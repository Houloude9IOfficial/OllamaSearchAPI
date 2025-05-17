
# 📚 Ollama Library API Proxy

A **simple proxy API** that fetches, parses, and **caches** data from [ollama.com](https://ollama.com/).

🔧 **Documentation**  
- [Swagger UI (/docs)](https://ollamasearchapi.onrender.com/docs)  
- [ReDoc UI (/redoc)](https://ollamasearchapi.onrender.com/redoc)

🕒 **Note:** All responses are cached for **6 hours** by default.

---

## 🔍 Example Endpoints

### 📊 Library Namespace
- 🔥 Popular models:  
  [`/library?o=popular`](https://ollamasearchapi.onrender.com/library?o=popular)
- 🆕 Newest models:  
  [`/library?o=newest`](https://ollamasearchapi.onrender.com/library?o=newest)
- 👁️ Filter by vision capability:  
  [`/library?c=vision`](https://ollamasearchapi.onrender.com/library?c=vision)

---

### 👤 User-Specific Queries
- 🔥 Popular models by `jmorganca`:  
  [`/jmorganca?o=popular`](https://ollamasearchapi.onrender.com/jmorganca?o=popular)
- 📄 Get details for `nextai` by `htdevs`:  
  [`/htdevs/nextai`](https://ollamasearchapi.onrender.com/htdevs/nextai)

---

### 🔎 Search & Model Info
- 🔍 Search for `mistral`:  
  [`/search?q=mistral`](https://ollamasearchapi.onrender.com/search?q=mistral)
- 📘 Details for `llama3`:  
  [`/library/llama3`](https://ollamasearchapi.onrender.com/library/llama3)
- 🏷️ Tag details for `llama3:8b`:  
  [`/library/llama3:8b`](https://ollamasearchapi.onrender.com/library/llama3:8b)
- 🏷️ All tags for `llama3`:  
  [`/library/llama3/tags`](https://ollamasearchapi.onrender.com/library/llama3/tags)

---

### 🧱 Blobs & Digests
- 📦 Get `model` blob for `llama3:8b`:  
  [`/library/llama3:8b/blobs/model`](https://ollamasearchapi.onrender.com/library/llama3:8b/blobs/model)
- ⚙️ Get `params` blob for `llama3:8b`:  
  [`/library/llama3:8b/blobs/params`](https://ollamasearchapi.onrender.com/library/llama3:8b/blobs/params)
- 🧬 Get blob by digest:  
  [`/library/llama3:8b/blobs/a3de86cd1c13`](https://ollamasearchapi.onrender.com/library/llama3:8b/blobs/a3de86cd1c13)

---

### 🧑 Credits

 - Programmed by [Blood Shot](https://discord.com/users/575254127748317194)   
 - Maintained by [Houloude9](https://discord.com/users/947432701160480828)   
 -   Hosted by [Render](https://render.com)

> 💡 **Tip:** Use these endpoints to build your own Ollama-powered dashboards, tools, or integrations effortlessly.