import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
from toolbox_core import ToolboxSyncClient

async def main():
    with ToolboxSyncClient("http://127.0.0.1:5000") as toolbox_client:
        
        prompt = """
        You are an assistant for an Amazon reviews Neo4j graph:

        Graph:
        (Customer)-[:WROTE]->(Review)-[:REVIEWS]->(Product)

        Properties:
        Customer.customer_id (int)
        Review.review_id, star_rating, review_headline, review_body, review_date
        Product.product_id, product_title, product_category

        Rules:
        - Prefer product_id over title search.
        - Convert customer_id to integer if needed.
        - Always limit to small result sets (max 5) to avoid overload.
        - Show concise outputs with rating, date, and short snippet.
        - Always mention which tool was used (e.g., 'via product-stats').
        """



        root_agent = Agent(
            model="gemini-1.5-flash",
            name="neo4jMagent",
            description="Movie assistant",
            instruction=prompt,
            tools=toolbox_client.load_toolset("my-toolset"),
        )
        session_service = InMemorySessionService()
        artifacts_service = InMemoryArtifactService()

        session = await session_service.create_session(
            state={}, app_name="neo4jMagent", user_id="123"
        )

        runner = Runner(
            app_name="neo4jMagent",
            agent=root_agent,
            artifact_service=artifacts_service,
            session_service=session_service,
        )

        queries = [
            "Use: top-products-apparel. Brief: top 3 Apparel items by review count.",
            "Use: top-products-automotive. Brief: top 3 Automotive items.",
            "Use: top-products-baby. Brief: top 3 Baby products.",
            "Use: top-products-beauty. Brief: top 3 Beauty products.",
            "Use: product-stats-demo. Brief: stats for B00992CF6W.",
            "Use: recent-reviews-demo. Brief: latest 3 reviews overall.",
            "Use: customer-recent-reviews. Brief: last 3 reviews from a given customer.",
            "Use: search-reviews-demo. Brief: find 3 reviews containing 'excellent'."
        ]


        for query in queries:
            content = types.Content(role="user", parts=[types.Part(text=query)])
            events = runner.run(
                session_id=session.id,
                user_id="123",
                new_message=content
            )

            responses = (
                part.text
                for event in events
                for part in event.content.parts
                if part.text is not None
            )

            for text in responses:
                print(text)

asyncio.run(main())
