`# Divergent Agent

A LangGraph agent with a Streamlit chat UI. The user's question is rephrased
into a fuller instruction, routed to one of four tools (Playwright, static
web scraper, Tavily search, or Qdrant RAG), and answered by a
user-selectable model — **Llama 3.3 70B or GPT-OSS 120B, both free models
served through OpenRouter.**

## Pipeline

```
query -> rephraser (Llama 3.3 70B :free) -> router (Llama 3.3 70B :free)
                                |
                +---------------+---------------+
                |         |            |         |
           playwright   scrape       search      rag
                |         |            |         |
                +---------------+---------------+
                                |
                    answering LLM (Llama 3.3 70B / GPT-OSS 120B, user's choice)
```

No cycling: each tool node runs once and feeds straight into the answering
LLM.

**Cost:** the rephraser, router, and both answer-model options run on
OpenRouter's `:free` tier — $0 per token, no credit card. OpenRouter caps
free models at roughly 20 requests/minute and 50/day (1,000/day once you've
ever added $10 in credits to the account, even if you don't spend it).
Free model IDs rotate over time — if either model in
`graph/nodes.py` (`ANSWER_MODEL_MAP`) or `graph/router.py` stops working,
check https://openrouter.ai/models?max_price=0 for a current replacement.
Tavily's free tier (1,000 credits/month) covers the search node. Qdrant is
free since it's your own local Docker container. The one piece that isn't
automatically free is the RAG embedding step — see the Qdrant note below.

## Setup

1. **Clone and enter the repo**

   ```bash
   git clone <your-repo-url>
   cd divergent-agent
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Configure environment variables**

   ```bash
   copy .env.example .env       # Windows
   # cp .env.example .env       # macOS/Linux
   ```

   Get a free `OPENROUTER_API_KEY` at https://openrouter.ai (no credit card
   needed) and fill in `TAVILY_API_KEY`. `OPENAI_API_KEY` is only needed if
   your existing Qdrant collection was embedded with OpenAI's
   `text-embedding-3-small` — see the note below.

4. **Start your existing Qdrant container**

   ```bash
   docker start divergent-qdrant
   docker ps   # confirm it's up on 0.0.0.0:6333-6334
   ```

   `QDRANT_URL` in `.env` defaults to `http://localhost:6333`, and
   `QDRANT_COLLECTION` defaults to `divergent_docs` — update it to match
   whatever collection you already indexed.

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Note on RAG embeddings and cost

`tools/qdrant_rag.py` currently uses `OpenAIEmbeddings("text-embedding-3-small")`
to embed each query before searching your collection — this is **not** free,
and more importantly, it only works correctly if that's the same embedding
model your existing `divergent-qdrant` collection was originally indexed
with (vector search requires query and stored vectors to come from the same
embedding model/dimension). If you embedded your documents with something
else, tell me what and I'll update this file to match — a free/local option
like `sentence-transformers` via `langchain-huggingface` is possible if the
collection supports it.

## Project structure

```
divergent-agent/
├── app.py                  # Streamlit UI + model picker
├── graph/
│   ├── state.py             # shared AgentState
│   ├── nodes.py              # rephraser, tool nodes, answering LLM
│   ├── router.py             # structured-output route decision
│   └── graph.py               # StateGraph wiring
├── tools/
│   ├── tavily_search.py
│   ├── qdrant_rag.py
│   ├── web_scraper.py       # requests + BeautifulSoup (static)
│   └── playwright_scraper.py # headless Chromium (dynamic/JS pages)
├── ui/
│   ├── components.py         # chat bubble render helpers
│   └── styles.css
├── requirements.txt
├── .env.example
└── .gitignore
```

## Creating the GitHub repo

This environment can't authenticate to your GitHub account, so create and
push the repo yourself:

```bash
cd divergent-agent
git init
git add .
git commit -m "Initial commit: LangGraph agent + Streamlit UI"

# Using GitHub CLI (recommended)
gh repo create divergent-agent --private --source=. --remote=origin --push

# Or manually: create an empty repo named divergent-agent on github.com,
# then:
git remote add origin https://github.com/<your-username>/divergent-agent.git
git branch -M main
git push -u origin main
```
