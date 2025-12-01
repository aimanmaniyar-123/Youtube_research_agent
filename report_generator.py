# report_generator.py
"""
ReportGenerator: calls LLMService (your services_llm_service.get_llm_service),
prepares final JSON structure and exports PDF/HTML strings.

Requirements:
  - uses your existing LLMService.get_llm_service()
  - uses simple text->pdf/html helpers (you can reuse earlier fpdf/docx code)
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from services_llm_service import get_llm_service
from utils_logger import get_logger

logger = get_logger("report_generator")

class ReportGenerator:
    def __init__(self, llm=None):
        self.llm = llm or get_llm_service()

    async def generate_report_json(
        self,
        channel_info: Dict[str, Any],
        videos: List[Dict[str, Any]],
        semantic_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Calls LLMService.generate_analysis() and returns the cleaned JSON.
        """
        try:
            summary = await self.llm.generate_analysis(
                channel_info=channel_info,
                videos=videos,
                semantic_texts=semantic_texts
            )
            # LLM returns JSON as per your schema. Validate minimal fields.
            if not isinstance(summary, dict):
                logger.error("LLM did not return dict for analysis")
                raise ValueError("Invalid LLM output")

            report = {
                "generated_at": datetime.utcnow().isoformat(),
                "channel": channel_info,
                "videos": videos,
                "analysis": summary
            }
            return report

        except Exception as e:
            logger.error(f"ReportGenerator.generate_report_json failed: {e}")
            # return fallback minimal report
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "channel": channel_info,
                "videos": videos,
                "analysis": {
                    "executive_summary": "Not Available",
                    "metrics": {},
                    "themes": [],
                    "insights": [],
                    "recommendations": []
                },
                "error": str(e)
            }

    def to_markdown(self, report_json: Dict[str, Any]) -> str:
        """Simple markdown conversion for PDF/HTML rendering."""
        analysis = report_json.get("analysis", {})
        md = []
        md.append(f"# YouTube Channel Analysis: {report_json['channel'].get('title', 'Unknown')}")
        md.append(f"Generated: {report_json.get('generated_at')}")
        md.append("\n## Executive Summary\n")
        md.append(analysis.get("executive_summary", "Not Available"))
        md.append("\n## Key Metrics\n")
        metrics = analysis.get("metrics", {})
        for k, v in metrics.items():
            md.append(f"- **{k}**: {v}")
        md.append("\n## Themes\n")
        themes = analysis.get("themes", [])
        for t in themes:
            md.append(f"- {t.get('name')} (freq: {t.get('frequency')}, engagement: {t.get('engagement')})")
        md.append("\n## Insights\n")
        for ins in analysis.get("insights", []):
            md.append(f"- {ins.get('text')} (confidence: {ins.get('confidence')})")
        md.append("\n## Recommendations\n")
        for rec in analysis.get("recommendations", []):
            md.append(f"- {rec.get('title')}: {rec.get('description')} (priority: {rec.get('priority')})")

        return "\n\n".join(md)

    def to_pdf_bytes(self, report_json: Dict[str, Any], text_to_pdf_bytes_fn) -> bytes:
        """
        Accepts a helper function to convert text->pdf bytes (e.g. your text_to_pdf_bytes).
        This avoids duplicating PDF generation logic.
        """
        md = self.to_markdown(report_json)
        return text_to_pdf_bytes_fn(md)

    def to_html_bytes(self, report_json: Dict[str, Any], title: Optional[str] = "YouTube Analysis") -> bytes:
        # Minimal HTML export
        html = f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>"
        html += f"<h1>{report_json['channel'].get('title','Channel')}</h1>"
        html += f"<p><em>Generated: {report_json.get('generated_at')}</em></p>"
        analysis = report_json.get("analysis", {})
        html += f"<h2>Executive Summary</h2><p>{analysis.get('executive_summary','Not Available')}</p>"
        html += "<h2>Metrics</h2><ul>"
        for k,v in analysis.get("metrics", {}).items():
            html += f"<li><strong>{k}:</strong> {v}</li>"
        html += "</ul>"
        html += "<h2>Themes</h2><ul>"
        for t in analysis.get("themes", []):
            html += f"<li>{t.get('name')} — frequency: {t.get('frequency')}</li>"
        html += "</ul>"
        html += "</body></html>"
        return html.encode("utf-8")
