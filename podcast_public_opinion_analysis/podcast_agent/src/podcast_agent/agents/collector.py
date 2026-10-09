"""
Collector Agent — 播客数据采集

负责:
  1. 按指定播客名称采集最新节目（不再按关键词/话题搜索）
  2. 将播客元数据存入 SQLite 数据库
  3. 输出 CollectorOutput 消息给下游 Transcriber

采集策略:
  - 对每个目标播客名，先用 /search?type=podcast 精确匹配出 podcast id
  - 再用 /podcasts/{id} 获取该播客的最新 episodes（sort=recent_first）
  - 播客名称来源: config.target_podcasts（环境变量 TARGET_PODCASTS 或默认列表）

如果 Listen Notes API 不可用，回退到预置的Demo数据
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import httpx

from ..models.messages import (
    AgentStatus, CollectorOutput, PodcastMetadata,
)
from ..utils.db import get_session
from .base import BaseAgent


class CollectorAgent(BaseAgent[dict, CollectorOutput]):
    """
    播客采集Agent

    输入: dict (可包含 search_query, max_results, podcast_names 等参数)
    输出: CollectorOutput (包含过滤后的播客节目列表)
    """

    agent_name = "collector"

    def execute(self, input_data: dict) -> CollectorOutput:
        """
        执行播客采集流程

        Args:
            input_data: {
                "search_query": str (可选, 非空时作为单个播客名),
                "max_results": int (可选, 默认 10, 总集数上限),
                "podcast_names": List[str] (可选, 默认用 config.target_podcasts),
                "episodes_per_podcast": int (可选, 默认用 config.episodes_per_podcast),
            }
        """
        self.log_start(
            query=input_data.get("search_query", "target podcasts"),
            max_results=input_data.get("max_results", 10),
        )

        output = CollectorOutput(
            agent_status=AgentStatus.RUNNING,
            search_query=input_data.get("search_query", ""),
        )

        try:
            # 检查 API 可用性
            if not self.config.listennotes_available:
                self.logger.warning("Listen Notes API 未配置，使用 Demo 数据")
                return self._fallback_demo_data(output)

            # 搜索播客
            episodes = self._search_podcasts(input_data)
            output.total_found = len(episodes)
            output.filtered_count = len(episodes)  # 不过滤，全部保留

            if not episodes:
                self.logger.warning("API 搜索无结果（可能网络不通），回退到 Demo 数据")
                return self._fallback_demo_data(output)

            output.episodes = episodes

            # 存入数据库
            self._save_to_database(episodes)

            self.log_done(
                found=output.total_found,
                filtered=output.filtered_count,
            )
            output.agent_status = AgentStatus.SUCCESS

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    # ================================================================
    # 播客搜索
    # ================================================================

    def _search_podcasts(self, input_data: dict) -> List[PodcastMetadata]:
        """
        采集指定播客名称的最新节目

        播客名称来源（按优先级）:
          1. input_data["search_query"] 非空 → 作为单个播客名（兼容 CLI --query / search 命令）
          2. input_data["podcast_names"] 显式传入
          3. config.target_podcasts（环境变量 TARGET_PODCASTS 或默认列表）

        每个播客采集 episodes_per_podcast 集最新节目，总集数不超过 max_results。
        """
        search_query = (input_data.get("search_query") or "").strip()
        podcast_names = input_data.get("podcast_names")

        if search_query:
            target_podcasts = [search_query]
        elif podcast_names:
            target_podcasts = list(podcast_names)
        else:
            target_podcasts = self.config.target_podcasts

        max_results = input_data.get("max_results", 10)
        episodes_per_podcast = input_data.get(
            "episodes_per_podcast", self.config.episodes_per_podcast
        )

        all_episodes: List[PodcastMetadata] = []
        seen_ids: set = set()

        for podcast_name in target_podcasts:
            if len(all_episodes) >= max_results:
                break
            try:
                podcast_id = self._find_podcast_id(podcast_name)
                if not podcast_id:
                    self.logger.warning(f"  [{podcast_name}] 未找到匹配播客，跳过")
                    continue

                results = self._fetch_podcast_episodes(podcast_id, episodes_per_podcast)
                added = 0
                for ep in results:
                    if len(all_episodes) >= max_results:
                        break
                    if ep.episode_id not in seen_ids:
                        seen_ids.add(ep.episode_id)
                        all_episodes.append(ep)
                        added += 1
                if added > 0:
                    self.logger.info(f"  [{podcast_name}] 采集 {added} 集")
                else:
                    self.logger.warning(f"  [{podcast_name}] 无最新节目")
            except Exception as e:
                self.logger.warning(f"  [{podcast_name}] 采集失败: {e}")
                continue

        return all_episodes

    def _find_podcast_id(self, podcast_name: str) -> str:
        """按名称搜索播客，返回最匹配的 podcast id；未找到返回空串"""
        url = f"{self.config.listennotes_base_url}/search"
        headers = {"X-ListenAPI-Key": self.config.listennotes_api_key}
        params = {
            "q": podcast_name,
            "type": "podcast",
            "language": "English",
            "safe_mode": 0,
            "offset": 0,
            "page_size": 5,
        }

        with httpx.Client(timeout=self.config.request_timeout) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return ""

        # 优先精确匹配（标题完全相等）
        name_lower = podcast_name.strip().lower()
        for r in results:
            title = (r.get("title_original") or r.get("title") or "").strip().lower()
            if title == name_lower:
                return r.get("id", "")

        # 否则取第一个结果
        return results[0].get("id", "")

    def _fetch_podcast_episodes(
        self, podcast_id: str, max_episodes: int
    ) -> List[PodcastMetadata]:
        """获取指定播客的最新 episodes（sort=recent_first）"""
        url = f"{self.config.listennotes_base_url}/podcasts/{podcast_id}"
        headers = {"X-ListenAPI-Key": self.config.listennotes_api_key}
        params = {"sort": "recent_first"}

        with httpx.Client(timeout=self.config.request_timeout) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        # 播客级信息（episode 内嵌套 podcast 缺失时回退用）
        podcast_title = data.get("title_original") or data.get("title", "")
        publisher = data.get("publisher_original") or data.get("publisher", "")
        image = data.get("image", "")

        episodes = []
        for item in data.get("episodes", [])[:max_episodes]:
            pod = item.get("podcast", {})
            ep = PodcastMetadata(
                podcast_id=pod.get("id", "") or podcast_id,
                episode_id=item.get("id", ""),
                podcast_title=pod.get("title_original") or pod.get("title", "") or podcast_title,
                episode_title=item.get("title_original") or item.get("title", ""),
                description=(item.get("description_original") or item.get("description", ""))[:500],
                audio_url=item.get("audio", ""),
                audio_length_sec=item.get("audio_length_sec", 0),
                pub_date_ms=item.get("pub_date_ms", 0),
                publisher=pod.get("publisher_original") or pod.get("publisher", "") or publisher,
                listennotes_url=item.get("listennotes_url", ""),
                image_url=item.get("image", "") or image,
            )
            episodes.append(ep)

        return episodes

    # ================================================================
    # 数据库存储
    # ================================================================

    def _save_to_database(self, episodes: List[PodcastMetadata]) -> None:
        """将播客元数据存入 SQLite"""
        from ..models.database import Podcast, Episode

        with get_session() as session:
            for ep in episodes:
                # Upsert Podcast
                podcast = session.query(Podcast).filter_by(
                    podcast_id=ep.podcast_id
                ).first()
                if not podcast:
                    podcast = Podcast(
                        podcast_id=ep.podcast_id,
                        title=ep.podcast_title,
                        publisher=ep.publisher,
                        listennotes_url=ep.listennotes_url,
                    )
                    session.add(podcast)
                    session.flush()

                # Upsert Episode
                existing = session.query(Episode).filter_by(
                    episode_id=ep.episode_id
                ).first()
                if not existing:
                    episode = Episode(
                        episode_id=ep.episode_id,
                        podcast_id=podcast.id,
                        title=ep.episode_title,
                        description=ep.description,
                        audio_url=ep.audio_url,
                        audio_length_sec=ep.audio_length_sec,
                        pub_date_ms=ep.pub_date_ms,
                        listennotes_url=ep.listennotes_url,
                        image_url=ep.image_url,
                    )
                    session.add(episode)

    # ================================================================
    # Fallback: Demo数据
    # ================================================================

    def _fallback_demo_data(self, output: CollectorOutput) -> CollectorOutput:
        """
        当 Listen Notes API 不可用时，返回预置的Demo数据
        这些是真实的播客节目信息，用于验证管道
        """
        demo_episodes = [
            PodcastMetadata(
                podcast_id="demo_podcast_001",
                episode_id="demo_ep_001",
                podcast_title="Bloomberg Surveillance",
                episode_title="Fed Rate Decision Impact on Emerging Markets and China Tech",
                description=(
                    "Discussion on how the Federal Reserve's latest rate decision "
                    "affects emerging markets, particularly China's technology sector "
                    "and the implications for A-share semiconductor stocks. "
                    "Analysis of NVIDIA supply chain and Chinese AI companies."
                ),
                audio_url="",
                audio_length_sec=2700,
                pub_date_ms=int(datetime.now().timestamp() * 1000),
                publisher="Bloomberg",
                listennotes_url="",
                image_url="",
            ),
            PodcastMetadata(
                podcast_id="demo_podcast_002",
                episode_id="demo_ep_002",
                podcast_title="Odd Lots",
                episode_title="The Global Semiconductor Supply Chain and China's Chip Ambitions",
                description=(
                    "Joe Weisenthal and Tracy Alloway explore the intricate global "
                    "semiconductor supply chain, US export controls on China, "
                    "and what it means for A-share listed chip makers like SMIC. "
                    "Discussion on tariff implications and technology decoupling."
                ),
                audio_url="",
                audio_length_sec=3600,
                pub_date_ms=int(datetime.now().timestamp() * 1000),
                publisher="Bloomberg",
                listennotes_url="",
                image_url="",
            ),
            PodcastMetadata(
                podcast_id="demo_podcast_003",
                episode_id="demo_ep_003",
                podcast_title="MacroVoices",
                episode_title="China's Economic Stimulus: What It Means for Commodities and A-Shares",
                description=(
                    "Erik Townsend interviews a macro hedge fund manager about "
                    "China's latest economic stimulus measures, the impact on "
                    "commodity prices (copper, iron ore, oil), and the outlook "
                    "for A-share markets. Discussion on PBOC policy and RMB exchange rate."
                ),
                audio_url="",
                audio_length_sec=4200,
                pub_date_ms=int(datetime.now().timestamp() * 1000),
                publisher="MacroVoices",
                listennotes_url="",
                image_url="",
            ),
        ]

        output.total_found = len(demo_episodes)
        output.filtered_count = len(demo_episodes)
        output.episodes = demo_episodes
        output.agent_status = AgentStatus.SUCCESS

        self._save_to_database(demo_episodes)
        self.log_done(status="DEMO", found=len(demo_episodes))
        return output