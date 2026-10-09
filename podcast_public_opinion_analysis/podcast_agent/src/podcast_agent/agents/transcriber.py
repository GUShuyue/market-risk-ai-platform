"""
Transcriber Agent — 语音转录

负责:
  1. 解析 Listen Notes 音频重定向 URL，获取真实 CDN 地址
  2. 上传本地 MP3 文件到 DashScope 文件服务
  3. 调用 DashScope qwen-audio-3.0-asr-flash-filetrans API 进行云端转录
  4. 解析转录结果，提取全文和句子分段
  5. 无音频或 API 失败时回退到 episode description

技术栈:
  - DashScope ASR API: qwen-audio-3.0-asr-flash-filetrans
  - 异步提交 + 轮询等待模式
  - 语言: 英文播客
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from ..config import get_config
from ..models.messages import (
    AgentStatus, CollectorOutput, EpisodeTranscription,
    PodcastMetadata, TranscriberOutput, TranscriptSegment,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)

# ASR 轮询参数（端点与模型见 config: asr_upload_url / asr_submit_url / asr_task_url / asr_model）


class TranscriberAgent(BaseAgent[CollectorOutput, TranscriberOutput]):
    """
    语音转录Agent

    输入: CollectorOutput (包含播客节目列表 + audio_url)
    输出: TranscriberOutput (包含转录文本 + 分段信息)

    如果 audio_url 不可用, 回退使用 episode description
    """

    agent_name = "transcriber"

    def __init__(self):
        super().__init__()
        self._http_client: Optional[httpx.Client] = None

    @property
    def http_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=self.config.request_timeout * 4,
                follow_redirects=True,
            )
        return self._http_client

    def execute(self, input_data: CollectorOutput) -> TranscriberOutput:
        self.log_start(episodes_count=len(input_data.episodes))

        output = TranscriberOutput(
            agent_status=AgentStatus.RUNNING,
        )

        all_segments: List[TranscriptSegment] = []
        all_texts: List[str] = []
        total_duration = 0.0
        episode_transcriptions: List[EpisodeTranscription] = []

        try:
            for i, ep in enumerate(input_data.episodes):
                self.logger.info(
                    f"  [{i+1}/{len(input_data.episodes)}] {ep.episode_title[:60]}"
                )

                if ep.audio_url:
                    segments, full_text, duration = self._transcribe_via_dashscope(ep)
                else:
                    segments, full_text, duration = self._transcribe_from_description(ep)

                all_segments.extend(segments)
                all_texts.append(full_text)
                total_duration += duration

                # 记录每集独立转录结果
                episode_transcriptions.append(EpisodeTranscription(
                    episode_id=ep.episode_id,
                    podcast_title=ep.podcast_title,
                    episode_title=ep.episode_title,
                    full_text=full_text,
                    segments=segments,
                    word_count=len(full_text.split()),
                    duration_seconds=duration,
                    description=ep.description or "",
                ))

            # 合并结果
            output.full_text = "\n\n".join(all_texts)
            output.segments = all_segments
            output.word_count = len(output.full_text.split())
            output.duration_seconds = total_duration
            output.language = self.config.asr_language
            output.episode_transcriptions = episode_transcriptions
            output.agent_status = AgentStatus.SUCCESS

            self.log_done(
                word_count=output.word_count,
                duration_s=f"{total_duration:.0f}s",
                segments=len(all_segments),
                episodes=len(episode_transcriptions),
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    # ================================================================
    # DashScope ASR 云端转录
    # ================================================================

    def _transcribe_via_dashscope(self, ep: PodcastMetadata) -> Tuple[List[TranscriptSegment], str, float]:
        """
        通过 DashScope ASR API 转录音频

        Returns:
            (segments, full_text, duration_seconds)
        """
        # 1. 解析重定向 URL，获取真实 CDN 地址
        self.logger.info(f"    解析音频 URL...")
        resolved_url = self._resolve_audio_url(ep.audio_url)
        if not resolved_url:
            self.logger.warning("    无法解析音频 URL，回退到 description")
            return self._transcribe_from_description(ep)

        self.logger.info(f"    CDN: {resolved_url[:80]}...")

        # 2. 提交 ASR 任务
        api_key = self.config.asr_api_key
        if not api_key:
            self.logger.warning("    ASR API Key 未配置，回退到 description")
            return self._transcribe_from_description(ep)

        task_id = self._submit_asr_task(api_key, resolved_url)
        if not task_id:
            self.logger.warning("    提交 ASR 任务失败，回退到 description")
            return self._transcribe_from_description(ep)

        # 3. 轮询等待任务完成
        transcription_url = self._poll_asr_task(api_key, task_id)
        if not transcription_url:
            self.logger.warning("    ASR 任务未完成，回退到 description")
            return self._transcribe_from_description(ep)

        # 4. 下载并解析转录结果
        segments, full_text, duration = self._parse_transcription(transcription_url)
        self.logger.info(
            f"    转录完成: {len(full_text)} 字符, {len(segments)} 段"
        )

        return segments, full_text, duration

    def _resolve_audio_url(self, url: str) -> Optional[str]:
        """解析 Listen Notes 重定向 URL，获取真实 CDN 地址"""
        try:
            resp = self.http_client.head(url)
            resp.raise_for_status()
            # httpx with follow_redirects=True 会自动跟随重定向
            # 最终 URL 就是 CDN 地址
            final_url = str(resp.url)
            return final_url
        except Exception as e:
            self.logger.warning(f"    解析 URL 失败: {e}")
            return None

    def _submit_asr_task(self, api_key: str, audio_url: str) -> Optional[str]:
        """提交 ASR 转录任务，返回 task_id"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        # oss:// URL 需要 X-DashScope-OssResourceResolve header
        if audio_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"

        payload = {
            "model": self.config.asr_model,
            "input": {"file_urls": [audio_url]},
            "parameters": {
                "channel_id": [0],
                "language_hints": [self.config.asr_language],
            },
        }

        try:
            resp = self.http_client.post(
                self.config.asr_submit_url,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            task_id = data.get("output", {}).get("task_id")
            if task_id:
                self.logger.info(f"    ASR 任务已提交: {task_id}")
            return task_id
        except Exception as e:
            self.logger.warning(f"    提交 ASR 任务失败: {e}")
            return None

    def _poll_asr_task(self, api_key: str, task_id: str) -> Optional[str]:
        """轮询 ASR 任务状态，返回 transcription_url"""
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        query_url = self.config.asr_task_url.format(task_id=task_id)

        start_time = time.time()
        while (time.time() - start_time) < self.config.asr_max_wait:
            status, url, err = self._query_asr_task(query_url, headers)
            if status == "succeeded":
                return url
            if status == "failed":
                self.logger.warning(f"    ASR 任务失败: {err or 'unknown'}")
                return None
            # running / 查询异常 → 继续等待
            time.sleep(self.config.asr_poll_interval)

        self.logger.warning(f"    ASR 任务超时 ({self.config.asr_max_wait}s)")
        return None

    def _query_asr_task(
        self, query_url: str, headers: dict
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        单次查询 ASR 任务状态（无 sleep，可被单任务轮询和多任务统一轮询复用）

        Returns:
            ("succeeded", transcription_url, None) |
            ("failed", None, 错误信息) |
            ("running", None, None)
        """
        try:
            resp = self.http_client.get(query_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            output = data.get("output", {})
            status = output.get("task_status", "UNKNOWN")

            if status == "SUCCEEDED":
                results = output.get("results", [])
                if results:
                    sub = results[0]
                    if sub.get("subtask_status") == "SUCCEEDED":
                        url = sub.get("transcription_url")
                        return "succeeded", url, None
                    return "failed", None, sub.get("message", "subtask failed")
                return "failed", None, "no results in response"

            elif status == "FAILED":
                return "failed", None, output.get("message", "task failed")

            else:  # PENDING / RUNNING / UNKNOWN
                return "running", None, None

        except Exception as e:
            # 查询异常不立即判失败，交由调用方按超时处理
            self.logger.warning(f"    查询 ASR 任务失败: {e}")
            return "running", None, None

    def _parse_transcription(
        self, transcription_url: str
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """下载并解析转录结果 JSON"""
        try:
            resp = self.http_client.get(transcription_url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.logger.warning(f"    下载转录结果失败: {e}")
            return [], "", 0.0

        segments: List[TranscriptSegment] = []
        all_texts: List[str] = []
        total_duration_ms = 0

        transcripts = data.get("transcripts", [])
        for transcript in transcripts:
            sentences = transcript.get("sentences", [])
            for sent in sentences:
                text = sent.get("text", "").strip()
                if not text:
                    continue

                begin_ms = sent.get("begin_time", 0)
                end_ms = sent.get("end_time", 0)

                segments.append(TranscriptSegment(
                    speaker=f"SPEAKER_{sent.get('speaker_id', 0)}",
                    start_time=begin_ms / 1000.0,
                    end_time=end_ms / 1000.0,
                    text=text,
                    segment_index=len(segments),
                ))
                all_texts.append(text)
                total_duration_ms = max(total_duration_ms, end_ms)

        full_text = " ".join(all_texts)
        duration = total_duration_ms / 1000.0
        return segments, full_text, duration

    # ================================================================
    # 回退：使用 description 作为转录文本
    # ================================================================

    def _transcribe_from_description(
        self, ep: PodcastMetadata
    ) -> Tuple[List[TranscriptSegment], str, float]:
        """使用 episode description 作为转录文本"""
        text = ep.description or (
            f"[No description available for: {ep.episode_title}]"
        )

        self.logger.info(f"    [回退] 使用 description ({len(text)} 字符)")

        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        segments = []
        for idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            segments.append(TranscriptSegment(
                speaker="NARRATOR",
                start_time=float(idx * 15),
                end_time=float((idx + 1) * 15),
                text=sent,
                segment_index=idx,
            ))

        word_count = len(text.split())
        estimated_duration = max(60.0, word_count / 150 * 60)

        return segments, text, estimated_duration

    # ================================================================
    # 本地文件转录（绕过 Listen Notes API）
    # ================================================================

    def transcribe_local_file(self, file_path: str) -> TranscriberOutput:
        """
        转录本地 MP3 文件（绕过 Listen Notes API）

        工作流:
          1. 上传本地 MP3 到 DashScope 文件服务
          2. 提交 ASR 转录任务
          3. 轮询等待完成
          4. 下载并解析转录结果
        """
        import os

        file_name = os.path.basename(file_path)
        self.log_start(file_path=file_name)

        output = TranscriberOutput(
            agent_status=AgentStatus.RUNNING,
        )

        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.logger.info(f"  本地文件: {file_name} ({file_size_mb:.1f} MB)")

            # 1. 上传到 DashScope
            self.logger.info("  上传到 DashScope 文件服务...")
            audio_url = self._upload_to_dashscope(file_path)
            if not audio_url:
                self.logger.warning("  上传失败，回退到本地文件名作为描述")
                return self._fallback_local_output(file_path, output)

            # 2. 提交 ASR 任务
            api_key = self.config.asr_api_key
            task_id = self._submit_asr_task(api_key, audio_url)
            if not task_id:
                self.logger.warning("  提交 ASR 任务失败")
                return self._fallback_local_output(file_path, output)

            # 3. 轮询等待
            transcription_url = self._poll_asr_task(api_key, task_id)
            if not transcription_url:
                self.logger.warning("  ASR 任务未完成")
                return self._fallback_local_output(file_path, output)

            # 4. 解析结果
            segments, full_text, duration = self._parse_transcription(transcription_url)
            self.logger.info(
                f"  转录完成: {len(full_text)} 字符, {len(segments)} 段, {duration:.0f}s"
            )

            output = self._build_local_success_output(file_path, segments, full_text, duration, output)
            self.log_done(
                word_count=output.word_count,
                duration_s=f"{duration:.0f}s",
                segments=len(segments),
                episodes=1,
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    # ================================================================
    # 多文件并行转录: 并发上传+提交 → 统一轮询 → 逐个解析
    # ================================================================

    def transcribe_local_files(self, file_paths: List[str]) -> dict:
        """
        并行转录多个本地 MP3 文件

        旧流程是逐文件「上传→提交→轮询到完成」再处理下一个，总耗时 = 各文件之和。
        新流程分三段:
          1. 并发上传 + 提交 ASR 任务（网络带宽瓶颈，asr_concurrency 并发）
          2. 统一轮询所有在途任务（每轮对每个任务查一次，间隔 asr_poll_interval）
          3. 任务完成后下载解析转录结果
        总耗时 ≈ 最慢的一个文件，而非全部之和。

        Returns:
            {file_path: TranscriberOutput}，失败的文件值为 FAILED 状态输出；
            上传/提交阶段就失败的文件不在 dict 中
        """
        api_key = self.config.asr_api_key
        results: dict = {}

        if not api_key:
            self.logger.warning("  ASR API Key 未配置，全部文件转录失败")
            return results

        max_workers = max(1, min(self.config.asr_concurrency, len(file_paths)))
        self.logger.info(
            f"  并发上传+提交 {len(file_paths)} 个文件 (x{max_workers} 并发)..."
        )
        start = time.time()

        # --- Phase 1: 并发上传 + 提交任务 ---
        pending: dict = {}  # file_path -> (task_id, query_url)
        headers = {"Authorization": f"Bearer {api_key}"}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._upload_and_submit, fp): fp
                for fp in file_paths
            }
            for fut in as_completed(futures):
                fp = futures[fut]
                task_id = fut.result()
                if task_id:
                    pending[fp] = (
                        task_id,
                        self.config.asr_task_url.format(task_id=task_id),
                    )
                else:
                    self.logger.warning(f"  上传/提交失败: {os.path.basename(fp)}")
                    # 不放入 results，由调用方按缺失处理

        if not pending:
            return results

        self.logger.info(
            f"  {len(pending)}/{len(file_paths)} 个任务已提交 "
            f"({time.time()-start:.1f}s)，开始统一轮询..."
        )

        # --- Phase 2+3: 统一轮询，完成一个解析一个 ---
        while pending and (time.time() - start) < self.config.asr_max_wait:
            for fp in list(pending):
                task_id, query_url = pending[fp]
                status, url, err = self._query_asr_task(query_url, headers)
                file_name = os.path.basename(fp)

                if status == "succeeded" and url:
                    del pending[fp]
                    segments, full_text, duration = self._parse_transcription(url)
                    self.logger.info(
                        f"  [ASR 完成] {file_name}: {len(full_text)} 字符, "
                        f"{len(segments)} 段 (累计 {time.time()-start:.1f}s)"
                    )
                    results[fp] = self._build_local_success_output(
                        fp, segments, full_text, duration,
                        TranscriberOutput(agent_status=AgentStatus.RUNNING),
                    )
                elif status == "failed":
                    del pending[fp]
                    self.logger.warning(f"  [ASR 失败] {file_name}: {err or 'unknown'}")

            if pending:
                time.sleep(self.config.asr_poll_interval)

        # 超时仍未完成的任务
        for fp in list(pending):
            self.logger.warning(
                f"  [ASR 超时] {os.path.basename(fp)} ({self.config.asr_max_wait}s)"
            )
        del pending

        self.logger.info(
            f"  批量转录完成: {len(results)}/{len(file_paths)} 个成功 "
            f"(总耗时 {time.time()-start:.1f}s)"
        )
        return results

    def _upload_and_submit(self, file_path: str) -> Optional[str]:
        """上传单个文件到 DashScope 并提交 ASR 任务，返回 task_id（失败返回 None）"""
        try:
            audio_url = self._upload_to_dashscope(file_path)
            if not audio_url:
                return None
            return self._submit_asr_task(self.config.asr_api_key, audio_url)
        except Exception as e:
            self.logger.warning(f"  上传/提交异常 ({os.path.basename(file_path)}): {e}")
            return None

    def _build_local_success_output(
        self,
        file_path: str,
        segments: List[TranscriptSegment],
        full_text: str,
        duration: float,
        output: TranscriberOutput,
    ) -> TranscriberOutput:
        """构建本地文件转录成功输出（单文件与批量路径共用）"""
        file_name = os.path.basename(file_path)
        output.full_text = full_text
        output.segments = segments
        output.word_count = len(full_text.split())
        output.duration_seconds = duration
        output.language = self.config.asr_language
        output.episode_transcriptions = [
            EpisodeTranscription(
                episode_id=file_name,
                podcast_title=os.path.splitext(file_name)[0],
                episode_title=file_name,
                full_text=full_text,
                segments=segments,
                word_count=len(full_text.split()),
                duration_seconds=duration,
                description="",
            )
        ]
        output.agent_status = AgentStatus.SUCCESS
        return output

    def _upload_to_dashscope(self, file_path: str) -> Optional[str]:
        """
        上传本地文件到 DashScope 文件服务，返回 oss:// URL

        两步流程:
          1. GET /api/v1/uploads?action=getPolicy&model=<model> → 获取 OSS 凭证
          2. POST 到 OSS upload_host → 上传文件
          3. 返回 oss:// 格式的 URL（配合 X-DashScope-OssResourceResolve 使用）
        """
        import requests as req

        api_key = self.config.asr_api_key
        if not api_key:
            self.logger.warning("  ASR API Key 未配置")
            return None

        file_name = Path(file_path).name
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        self.logger.info(f"  上传文件: {file_name} ({file_size_mb:.1f} MB)")

        try:
            # Step 1: 获取 OSS 上传凭证
            self.logger.info("  获取 OSS 上传凭证...")
            cred_resp = req.get(
                self.config.asr_upload_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                params={
                    "action": "getPolicy",
                    "model": self.config.asr_model,
                },
                timeout=30,
            )
            cred_resp.raise_for_status()
            cred_data = cred_resp.json()["data"]

            upload_host = cred_data["upload_host"]
            upload_dir = cred_data["upload_dir"]
            policy = cred_data["policy"]
            signature = cred_data["signature"]
            oss_access_key_id = cred_data["oss_access_key_id"]
            x_oss_object_acl = cred_data.get("x_oss_object_acl", "private")
            x_oss_forbid_overwrite = cred_data.get("x_oss_forbid_overwrite", "true")

            self.logger.info(f"  凭证获取成功, 上传目录: {upload_dir}")

            # Step 2: 上传文件到 OSS
            self.logger.info(f"  上传到 OSS: {upload_host}...")
            with open(file_path, "rb") as f:
                oss_resp = req.post(
                    upload_host,
                    data={
                        "key": f"{upload_dir}/{file_name}",
                        "policy": policy,
                        "OSSAccessKeyId": oss_access_key_id,
                        "signature": signature,
                        "success_action_status": "200",
                        "x-oss-object-acl": x_oss_object_acl,
                        "x-oss-forbid-overwrite": x_oss_forbid_overwrite,
                    },
                    files={"file": (file_name, f, "audio/mpeg")},
                    timeout=600,
                )
                oss_resp.raise_for_status()

            # Step 3: 构建 oss:// URL
            oss_url = f"oss://{upload_dir}/{file_name}"
            self.logger.info(f"  上传成功: {oss_url}")

            return oss_url

        except Exception as e:
            self.logger.warning(f"  文件上传失败: {e}")
            return None

    def _fallback_local_output(
        self, file_path: str, output: TranscriberOutput
    ) -> TranscriberOutput:
        """本地文件转录失败时的回退输出"""
        import os
        file_name = os.path.basename(file_path)
        fallback_text = f"[Failed to transcribe local file: {file_name}]"

        output.full_text = fallback_text
        output.segments = []
        output.word_count = 0
        output.duration_seconds = 0
        output.language = self.config.asr_language
        output.episode_transcriptions = [
            EpisodeTranscription(
                episode_id=file_name,
                podcast_title=os.path.splitext(file_name)[0],
                episode_title=file_name,
                full_text=fallback_text,
                segments=[],
                word_count=0,
                duration_seconds=0,
                description="",
            )
        ]
        output.agent_status = AgentStatus.FAILED
        output.error_message = "Local file transcription failed"
        return output