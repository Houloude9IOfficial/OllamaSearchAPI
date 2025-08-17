# 📚 Ollama Library API Proxy

A **simple proxy API** that fetches, parses, and **caches** data from [ollama.com](https://ollama.com/).  

🔧 **Documentation**  
- [Swagger UI (/docs)](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/docs)  
- [ReDoc UI (/redoc)](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/redoc)  

🕒 **Note:** All responses are cached for **6 hours** by default.  

---

## Setup Instructions

1. **Rename the environment file**  
   Rename `example.env` to `.env`.

2. **Configure your environment**  
   Open the `.env` file and customize the values as needed, such as:
   - API URL
   - Cache duration
   - Enable static website mode

3. **Install dependencies**  
   Run the following command to install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the API**
   ```bash 
   ollama_library_api.py
   ```
   Or if you're getting errors:
   ```bash
   uvicorn ollama_library_api:app --host 0.0.0.0 --port 5115 --reload
   ``` 

### Example `.env`
```env
# CODE version of the app
CODE_VERSION=1.3.0_Release

# Base URL of Ollama
OLLAMA_COM_BASE_URL=https://ollama.com

# Your API URL
CURRENT_BASE_URL=https://example.com

# If running a static website
STATIC_WEBSITE=False

# Cache duration in hours
CACHE_EXPIRE_AFTER=6
```

## 🔍 Example Endpoints

### 📊 Library Namespace
- 🔥 Popular models:  
  [`/library?o=popular`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library?o=popular)
- 🆕 Newest models:  
  [`/library?o=newest`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library?o=newest)
- 👁️ Filter by vision capability:  
  [`/library?c=vision`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library?c=vision)

---

### 👤 User-Specific Queries
- 🔥 Popular models by `jmorganca`:  
  [`/jmorganca?o=popular`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/jmorganca?o=popular)
- 📄 Get details for `nextai` by `htdevs`:  
  [`/htdevs/nextai`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/htdevs/nextai)

---

### 🔎 Search & Model Info
- 🔍 Search for `mistral`:  
  [`/search?q=mistral`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/search?q=mistral)
- 📘 Details for `llama3`:  
  [`/library/llama3`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library/llama3)
- 🏷️ Tag details for `llama3:8b`:  
  [`/library/llama3:8b`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library/llama3:8b)
- 🏷️ All tags for `llama3`:  
  [`/library/llama3/tags`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library/llama3/tags)

---

### 🧱 Blobs & Digests
- 📦 Get `model` blob for `llama3:8b`:  
  [`/library/llama3:8b/blobs/model`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library/llama3:8b/blobs/model)
- ⚙️ Get `params` blob for `llama3:8b`:  
  [`/library/llama3:8b/blobs/params`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library/llama3:8b/blobs/params)
- 🧬 Get blob by digest:  
  [`/library/llama3:8b/blobs/a3de86cd1c13`](https://dear-franky-htdevssss-83f0aa7d.koyeb.app/library/llama3:8b/blobs/a3de86cd1c13)

---

### 🧑 Credits

 - Programmed by [Blood Shot](https://discord.com/users/575254127748317194)   
 - Maintained by [Houloude9](https://discord.com/users/947432701160480828)   
 -   Hosted by [Render](https://render.com)

> 💡 **Tip:** Use these endpoints to build your own Ollama-powered dashboards, tools, or integrations effortlessly.