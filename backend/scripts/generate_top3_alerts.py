"""Generate rich informative alert notifications for Top 3 AI-scored stocks."""

import asyncio
from datetime import UTC, datetime
from sqlalchemy import text
from app.db.database import get_db
from app.agents.alert import AlertAgent

TOP_STOCKS = [
    {
        "ticker": "BMRI",
        "name": "PT Bank Mandiri (Persero) Tbk",
        "alert_type": "price_spike",
        "severity": "high",
        "score": 70.5,
        "recommendation": "BUY",
        "price": 4030,
        "change": 0.052,
        "direction": "up",
        "context": "Ranked #1 Invelio Top Pick with 70.5 score. Low forward P/E 6.07x, high dividend yield >5%, and executive insider buying.",
    },
    {
        "ticker": "BBRI",
        "name": "PT Bank Rakyat Indonesia (Persero) Tbk",
        "alert_type": "volume_surge",
        "severity": "medium",
        "score": 70.4,
        "recommendation": "BUY",
        "price": 3080,
        "volume": 182450000,
        "context": "Ranked #2 Invelio Top Pick with 70.4 score. Attractive dividend yield >5.5%, institutional volume absorption, and micro-loan expansion.",
    },
    {
        "ticker": "ASII",
        "name": "Astra International Tbk",
        "alert_type": "price_spike",
        "severity": "high",
        "score": 68.1,
        "recommendation": "BUY",
        "price": 4630,
        "change": 0.0266,
        "direction": "up",
        "context": "Ranked #3 Invelio Top Pick with 68.1 score. Strong multi-sector holding breakout, foreign capital net buy, and automotive recovery.",
    },
]

async def main():
    async for db in get_db():
        agent = AlertAgent(db=db)
        now = datetime.now(UTC)
        created_alerts = []

        for stock in TOP_STOCKS:
            if stock["alert_type"] == "price_spike":
                details = {
                    "direction": stock["direction"],
                    "change": stock["change"],
                    "price": stock["price"],
                    "context": stock["context"],
                }
            else:
                details = {
                    "volume": stock["volume"],
                    "price": stock["price"],
                    "context": stock["context"],
                }

            msg = await agent._synthesize_alert_message(
                ticker=stock["ticker"],
                name=stock["name"],
                alert_type=stock["alert_type"],
                severity=stock["severity"],
                context_data=details,
            )

            # Insert into alerts table
            await db.execute(
                text(
                    "INSERT INTO alerts (ticker, alert_type, severity, message, is_read, created_at) "
                    "VALUES (:ticker, :alert_type, :severity, :message, FALSE, :now)"
                ),
                {
                    "ticker": stock["ticker"],
                    "alert_type": stock["alert_type"],
                    "severity": stock["severity"],
                    "message": msg,
                    "now": now,
                },
            )
            created_alerts.append({
                "ticker": stock["ticker"],
                "type": stock["alert_type"],
                "severity": stock["severity"],
                "message": msg,
            })

        await db.commit()
        print(f"Successfully generated and inserted {len(created_alerts)} notifications!")
        for idx, a in enumerate(created_alerts, 1):
            print(f"\n--- [Alert #{idx}: {a['ticker']} ({a['type']})] ---")
            print(a["message"])

if __name__ == "__main__":
    asyncio.run(main())
