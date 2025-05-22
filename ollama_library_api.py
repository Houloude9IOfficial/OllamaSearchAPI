import json
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request, Query, Path as FastApiPath
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse # Import FileResponse and HTMLResponse
from pydantic import BaseModel, Field, HttpUrl
import requests
import requests_cache
from datetime import datetime, timezone, timedelta
import uvicorn
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import os
import random
import time
from collections import deque

# --- Configuration ---

CODE_VERSION = '1.3.0_Release'
OLLAMA_COM_BASE_URL = "https://ollama.com"
CURRENT_BASE_URL = "https://ollamasearchapi.onrender.com"
STATIC_WEBSITE = False # RECOMMENDED "FALSE"
CACHE_EXPIRE_AFTER = 6 # HOURS

# --- Configuration ---


# Cache settings: SQLite backend, expires after 6 hours (21600 seconds)
# Use a simple in-memory cache for demonstration/testing, persistent 'sqlite' is also an option
cachetime = CACHE_EXPIRE_AFTER * 3600
requests_cache.install_cache('ollama_com_cache', backend='memory', expire_after=cachetime)
# Use a CachedSession for all requests to ollama.com
cached_session = requests_cache.CachedSession()


# --- DO NOT CHANGE WITHOUT KNOWLEDGE ---

Static_Website = STATIC_WEBSITE
filename = 'index.html'
format = 'html'
TEMPTEXT = f'{int(time.time() * 1000)}{format}'
FILETEMPTEXT = f'tempfile_{TEMPTEXT}.html'

if Static_Website == True:
    TEMPTEXT = f'index'
    FILETEMPTEXT = f'index.html'
# --- Pydantic Models ---

class CacheInfoMixin(BaseModel):
    fetched_at: datetime = Field(description="Timestamp when the data was fetched (or revalidated from cache).")
    cached_at: Optional[datetime] = Field(None, description="Timestamp when the data was originally cached. None if fresh fetch.")
    cache_expires_at: Optional[datetime] = Field(None, description="Timestamp when the cache for this item is set to expire.")
    from_cache: bool = Field(description="Indicates if the response was served from cache.")

class FilterInfo(BaseModel):
    capabilities: Optional[List[str]] = None

class ModelResultItem(BaseModel):
    source_url: HttpUrl
    namespace: str
    model_base_name: str
    name_full_model: str
    description: str
    pull_count_str: str
    pull_count: int
    tags_count: int
    last_updated_str: str
    last_updated_iso: datetime
    capabilities: List[str] = []
    sizes: List[str] = []

class SearchResponse(CacheInfoMixin):
    query: Optional[str] = None
    sort_order: str
    filters: Optional[FilterInfo] = None
    results: List[ModelResultItem]

class ModelListByNamespaceResponse(CacheInfoMixin): # Renamed from LibraryListResponse
    queried_namespace: str
    sort_order: str
    filters: Optional[FilterInfo] = None
    results: List[ModelResultItem]

class FileSummary(BaseModel):
    name: str
    blob_url: HttpUrl
    digest: Optional[str] = None
    size_str: str
    snippet: str
    updated_str: Optional[str] = None

class TagSummary(BaseModel):
    tag_part: str
    name_full_tag: str
    size_str: Optional[str] = None
    is_active: bool = False

class ModelPageResponse(CacheInfoMixin):
    name_full_model: str
    namespace: str
    model_base_name: str
    active_tag_part: Optional[str] = None
    active_tag_full_name: Optional[str] = None
    source_url: HttpUrl
    summary: str
    pull_count_str: str
    pull_count: int
    last_updated_str: str
    last_updated_iso: datetime
    capabilities: List[str] = []
    sizes: List[str] = []
    readme_content: str
    tag_command: Optional[str] = None
    tag_files_summary: List[FileSummary] = []
    all_tags_dropdown_summary: List[TagSummary] = []
    all_tags_page_url: HttpUrl
    total_tags_count_from_link: Optional[int] = None

class TagDetailItem(BaseModel):
    name_full_tag: str
    tag_part: str
    source_url: HttpUrl
    digest: str
    size_str: str
    size_bytes: int
    context_window_str: Optional[str] = None
    input_type: Optional[str] = None
    modified_str: str
    modified_iso: datetime
    is_default: bool = False

class AllTagsResponse(CacheInfoMixin):
    name_full_model: str
    namespace: str
    model_base_name: str
    tags_page_url: HttpUrl
    tags: List[TagDetailItem]

class GGUFMetadata(BaseModel):
    arch: Optional[str] = None
    parameters: Optional[str] = None
    quantization: Optional[str] = None
    class Config:
        extra = "allow"

class BlobDetailsResponse(CacheInfoMixin):
    name_full_tag: str
    canonical_name: str
    source_url: HttpUrl
    digest: str
    size_str: str
    text_content: Optional[str] = None
    parsed_json_content: Optional[Union[Dict[str, Any], List[Any]]] = None
    gguf_metadata_snippet: Optional[str] = None
    parsed_gguf_metadata: Optional[GGUFMetadata] = None
    listing_updated_str: Optional[str] = None
    listing_updated_iso: Optional[datetime] = None


# --- FastAPI App ---
app = FastAPI(
    title="Ollama.com Library API Proxy",
    description="An API that fetches, parses, and caches data from ollama.com.",
    version=CODE_VERSION, # Version increment
)

# --- Helper Functions ---

def get_cache_info_from_response(response: requests.Response) -> Dict[str, Any]:
    """Gets cache information from a requests-cache response."""
    cached_at = None
    expires_at = None

    if hasattr(response, 'created_at') and response.created_at: # type: ignore
        cached_at_naive = response.created_at # type: ignore
        cached_at = cached_at_naive.replace(tzinfo=timezone.utc) if cached_at_naive else None

    if hasattr(response, 'expires') and response.expires: # type: ignore
        expires_at_naive = response.expires # type: ignore
        expires_at = expires_at_naive.replace(tzinfo=timezone.utc) if expires_at_naive else None
    elif cached_at: # If created_at exists but expires is None, calculate based on default
        default_expiry_seconds = 21600 # Default from install_cache
        if hasattr(requests_cache.get_cache(), 'settings') and requests_cache.get_cache().settings.expire_after is not None: # type: ignore
             default_expiry_seconds = requests_cache.get_cache().settings.expire_after # type: ignore
        expires_at = cached_at + timedelta(seconds=default_expiry_seconds)

    return {
        "fetched_at": datetime.now(timezone.utc),
        "cached_at": cached_at,
        "cache_expires_at": expires_at,
        "from_cache": getattr(response, 'from_cache', False)
    }

def make_full_model_name(namespace: str, model_base_name: str) -> str:
    if namespace == "library":
        return model_base_name
    return f"{namespace}/{model_base_name}"

def make_full_tag_name(namespace: str, model_base_name: str, tag_part: str) -> str:
    full_model_name = make_full_model_name(namespace, model_base_name)
    return f"{full_model_name}:{tag_part}"

def parse_pull_count(pull_str: str) -> int:
    pull_str = pull_str.lower().replace(',', '').strip()
    if not pull_str: return 0
    if 'm' in pull_str:
        return int(float(pull_str.replace('m', '')) * 1_000_000)
    elif 'k' in pull_str:
        return int(float(pull_str.replace('k', '')) * 1_000)
    try:
        return int(pull_str)
    except ValueError:
        return 0

def parse_relative_date_to_datetime(relative_str: str, base_time: datetime = datetime.now(timezone.utc)) -> datetime:
    relative_str = relative_str.lower().strip()
    if "just now" in relative_str or "moments ago" in relative_str:
        return base_time
    if "yesterday" in relative_str:
        # Set time to midnight of yesterday
        return base_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

    match = re.match(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", relative_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "minute": return base_time - timedelta(minutes=value)
        if unit == "hour": return base_time - timedelta(hours=value)
        if unit == "day": return base_time - timedelta(days=value)
        if unit == "week": return base_time - timedelta(weeks=value)
        # Approximation for month/year
        if unit == "month": return base_time - timedelta(days=value * 30)
        if unit == "year": return base_time - timedelta(days=value * 365)

    print(f"Warning: Could not parse relative date '{relative_str}'. Returning base_time.")
    return base_time

def parse_ollama_absolute_date_str(date_str: str) -> datetime:
    try:
        dt_naive = datetime.strptime(date_str.replace(" UTC", "").strip(), '%b %d, %Y %I:%M %p')
        return dt_naive.replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"Warning: Could not parse absolute date string '{date_str}': {e}. Using current time.")
        return datetime.now(timezone.utc)

def parse_size_str_to_bytes(size_str: str) -> int:
    if not size_str: return 0
    size_str_upper = size_str.upper().strip()

    val_str = re.sub(r"[^0-9.]", "", size_str)
    if not val_str: return 0
    try:
        val = float(val_str)
    except ValueError:
        return 0

    if "KB" in size_str_upper or "KIB" in size_str_upper: return int(val * 1024)
    if "MB" in size_str_upper or "MIB" in size_str_upper: return int(val * 1024 * 1024)
    if "GB" in size_str_upper or "GIB" in size_str_upper: return int(val * 1024 * 1024 * 1024)
    if "TB" in size_str_upper or "TIB" in size_str_upper: return int(val * 1024 * 1024 * 1024 * 1024)
    # Assuming plain number or Bytes (B) is bytes
    try:
        return int(val)
    except ValueError:
        return 0

# --- HTML Parsing Functions ---

def parse_model_listing_item(item_li: BeautifulSoup, base_url: str) -> Optional[Dict[str, Any]]:
    anchor = item_li.select_one('a[href]')
    if not anchor: return None

    raw_source_url = anchor['href']
    source_url = urljoin(base_url, raw_source_url)

    path_parts = urlparse(source_url).path.strip('/').split('/')
    namespace = "library" # Default
    model_base_name = ""

    if len(path_parts) >= 2:
        if path_parts[0].lower() == 'library' and len(path_parts) > 1:
            namespace = 'library'
            model_base_name = path_parts[1].lower()
        elif path_parts[0].lower() != 'library': # User namespace
            namespace = path_parts[0].lower()
            model_base_name = path_parts[1].lower()

    # Robust parsing of model name from title or URL
    title_h2_span = item_li.select_one('h2 span[x-test-search-response-title]')
    if title_h2_span:
        full_name_from_title = title_h2_span.get_text(strip=True)
        if '/' in full_name_from_title:
            ns_from_title, mbn_from_title = full_name_from_title.split('/',1)
            namespace = ns_from_title.lower()
            model_base_name = mbn_from_title.lower()
        elif namespace == "library": # If it's a library model, title is just base_name
            model_base_name = full_name_from_title.lower()

    if not model_base_name or (namespace != "library" and '/' not in item_li.select_one('h2').get_text()): # Refine fallback check
        title_div = item_li.select_one('div[x-test-model-title]')
        if title_div and title_div.has_attr('title'):
             full_name_from_title_attr = title_div['title'].lower()
             if '/' in full_name_from_title_attr :
                 namespace, model_base_name = full_name_from_title_attr.split('/',1)
             else:
                 model_base_name = full_name_from_title_attr


    if not model_base_name:
        return None

    description_p = item_li.select_one('p.max-w-lg.break-words')
    description = description_p.get_text(separator=" ", strip=True) if description_p else ""

    pull_count_span = item_li.select_one('span[x-test-pull-count]')
    pull_count_str = pull_count_span.get_text(strip=True) if pull_count_span else "0"
    pull_count = parse_pull_count(pull_count_str)

    tags_count_span = item_li.select_one('span[x-test-tag-count]')
    tags_count = int(tags_count_span.get_text(strip=True)) if tags_count_span and tags_count_span.get_text(strip=True).isdigit() else 0

    last_updated_iso_datetime = datetime.now(timezone.utc)
    last_updated_str = ""
    updated_span_el = item_li.select_one('span[x-test-updated]')
    if updated_span_el:
        last_updated_str = updated_span_el.get_text(strip=True)
        parent_title_span = updated_span_el.find_parent('span', title=True)
        if parent_title_span and parent_title_span['title']:
            try:
                last_updated_iso_datetime = parse_ollama_absolute_date_str(parent_title_span['title'])
            except ValueError:
                last_updated_iso_datetime = parse_relative_date_to_datetime(last_updated_str)
        else:
             last_updated_iso_datetime = parse_relative_date_to_datetime(last_updated_str)

    capabilities = [cap.get_text(strip=True).lower() for cap in item_li.select('span[x-test-capability]')]
    sizes = [size.get_text(strip=True).lower() for size in item_li.select('span[x-test-size]')]

    return {
        "source_url": source_url,
        "namespace": namespace,
        "model_base_name": model_base_name,
        "name_full_model": make_full_model_name(namespace, model_base_name),
        "description": description,
        "pull_count_str": pull_count_str,
        "pull_count": pull_count,
        "tags_count": tags_count,
        "last_updated_str": last_updated_str,
        "last_updated_iso": last_updated_iso_datetime.isoformat(),
        "capabilities": capabilities,
        "sizes": sizes,
    }

def parse_list_or_search_page_html(html_content: str, base_url: str = OLLAMA_COM_BASE_URL) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_content, 'html.parser')
    model_items_data = []
    list_items = soup.select('ul[role="list"] li[x-test-model]')
    for item_li in list_items:
        parsed_item = parse_model_listing_item(item_li, base_url)
        if parsed_item:
            model_items_data.append(parsed_item)
    return model_items_data

def parse_model_page_html(html_content: str, page_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_content, 'html.parser')

    path_parts = urlparse(page_url).path.strip('/').split('/')
    namespace = "library"
    model_base_name_from_url = ""
    active_tag_part_from_url = None

    if len(path_parts) >= 2:
        ns_or_model = path_parts[0].lower()
        model_or_tag = path_parts[1].lower()

        if len(path_parts) >= 3 and path_parts[1].lower() != 'tags':
             namespace = ns_or_model
             model_tag_combo = model_or_tag
             if ':' in model_tag_combo:
                 model_base_name_from_url, active_tag_part_from_url = model_tag_combo.split(':', 1)
             else:
                 model_base_name_from_url = model_tag_combo
        elif len(path_parts) >= 2 and path_parts[0].lower() == 'library':
             namespace = 'library'
             model_tag_combo = model_or_tag
             if ':' in model_tag_combo:
                 model_base_name_from_url, active_tag_part_from_url = model_tag_combo.split(':', 1)
             else:
                 model_base_name_from_url = model_tag_combo
        elif len(path_parts) >=2 and path_parts[0].lower() != 'library': # user/model (no tag)
            namespace = ns_or_model
            model_base_name_from_url = model_or_tag


    model_name_a = soup.select_one('a[x-test-model-name][title]')
    model_base_name = model_name_a['title'].lower() if model_name_a else model_base_name_from_url

    if namespace == 'library' and model_name_a and '/' in model_name_a['title']:
         ns_from_title, mbn_from_title = model_name_a['title'].lower().split('/', 1)
         namespace = ns_from_title
         model_base_name = mbn_from_title
    elif namespace != 'library' and model_name_a and '/' not in model_name_a['title'] and not model_base_name:
         model_base_name = model_name_a['title'].lower() # Case: namespace from URL, model base name from title
    elif not model_base_name and model_name_a: # General fallback if model_base_name is still not set
         model_base_name = model_name_a['title'].lower()


    summary_span = soup.select_one('#summary-content span, #summary-content')
    summary = summary_span.get_text(separator=" ", strip=True) if summary_span else "Summary not found."
    if not summary.strip() or summary.strip().lower() == "no summary":
        summary_textarea = soup.select_one('#summary-textarea')
        if summary_textarea:
            summary = summary_textarea.get_text(separator=" ", strip=True)


    pull_count_span = soup.select_one('span[x-test-pull-count]')
    pull_count_str = pull_count_span.get_text(strip=True) if pull_count_span else "0"
    pull_count = parse_pull_count(pull_count_str)

    updated_span_relative = soup.select_one('span[x-test-updated]')
    last_updated_str = ""
    last_updated_iso = datetime.now(timezone.utc)

    if updated_span_relative:
        last_updated_str = updated_span_relative.get_text(strip=True)
        parent_title_span = updated_span_relative.find_parent('span', title=True)
        if parent_title_span and parent_title_span['title']:
            try:
                last_updated_iso = parse_ollama_absolute_date_str(parent_title_span['title'])
            except ValueError: last_updated_iso = parse_relative_date_to_datetime(last_updated_str)
        else: last_updated_iso = parse_relative_date_to_datetime(last_updated_str)

    capabilities = [cap.get_text(strip=True).lower() for cap in soup.select('div.flex-wrap span.bg-indigo-50')]
    sizes = [size.get_text(strip=True).lower() for size in soup.select('span[x-test-size]')]

    tag_selection_section = soup.select_one('section[x-test-model-tag-selection]')
    active_tag_part = active_tag_part_from_url
    tag_command = None

    if tag_selection_section:
        active_tag_button_div = tag_selection_section.select_one('button[name="tag"] div.truncate')
        if active_tag_button_div and not active_tag_part:
            active_tag_part = active_tag_button_div.get_text(strip=True).lower()

        command_input = tag_selection_section.select_one('input.command[name="command"]')
        if command_input:
            tag_command = command_input['value']
            if not active_tag_part and tag_command and ":" in tag_command:
                 active_tag_part = tag_command.split(":")[-1].lower()
            elif not active_tag_part and tag_command and model_base_name in tag_command:
                run_command_parts = tag_command.strip().split()
                expected_full_model_name = make_full_model_name(namespace, model_base_name)
                if len(run_command_parts) == 3 and run_command_parts[0] == 'ollama' and run_command_parts[1] == 'run' and run_command_parts[2] == expected_full_model_name:
                     active_tag_part = "latest"


    tag_files_summary = []
    file_explorer_section = soup.select_one('#file-explorer section')
    if file_explorer_section:
        listing_updated_str_fe = ""
        #listing_updated_iso_fe = None # Not used for now
        updated_p_fe = file_explorer_section.select_one('div.bg-neutral-50 > p:first-of-type')
        if updated_p_fe:
             listing_updated_str_raw_fe = updated_p_fe.get_text(strip=True)
             if "Updated" in listing_updated_str_raw_fe:
                 listing_updated_str_fe = listing_updated_str_raw_fe.replace("Updated","").strip()
             elif re.match(r"\d+ \w+ ago", listing_updated_str_raw_fe): # Handles "X days ago"
                 listing_updated_str_fe = listing_updated_str_raw_fe


        for file_a in file_explorer_section.select('a.group.block.grid-cols-12'):
            name_div = file_a.select_one('div.sm\\:col-span-2')
            name = name_div.get_text(strip=True).lower() if name_div else "unknown"

            blob_url_href = file_a['href']
            blob_url = urljoin(OLLAMA_COM_BASE_URL, blob_url_href)

            url_path_parts = urlparse(blob_url).path.strip('/').split('/')
            digest_from_url = None
            if len(url_path_parts) > 1 and url_path_parts[-2] == "blobs":
                digest_from_url = url_path_parts[-1]
                if not re.match(r"^[0-9a-fA-F]{12,}$", digest_from_url): digest_from_url = None


            size_div = file_a.select_one('div.sm\\:col-start-12')
            size_str = size_div.get_text(strip=True) if size_div else "0B"

            snippet_div = file_a.select_one('div.sm\\:col-span-8')
            snippet = snippet_div.get_text(separator=" ", strip=True) if snippet_div else ""

            tag_files_summary.append(FileSummary(
                name=name, blob_url=blob_url, digest=digest_from_url,
                size_str=size_str, snippet=snippet, updated_str=listing_updated_str_fe
            ))

    readme_div = soup.select_one('#readme #display')
    readme_content = str(readme_div) if readme_div else "<p>Readme not found.</p>"

    all_tags_dropdown_summary = []
    tags_nav = soup.select_one('#tags-nav')
    if tags_nav:
        for tag_a_dropdown in tags_nav.select(f'a[href^="/library/"], a[href^="/{namespace}/"]'):
            if "View all" in tag_a_dropdown.get_text(): continue

            tag_name_span = tag_a_dropdown.select_one('span.truncate span.group-hover\\:underline')
            tag_part_from_dropdown = tag_name_span.get_text(strip=True).lower() if tag_name_span else ""

            size_span_dropdown = tag_a_dropdown.select_one('span.text-xs.text-neutral-400')
            size_str_from_dropdown = size_span_dropdown.get_text(strip=True) if size_span_dropdown else None

            is_active_tag_dropdown = ('bg-neutral-100' in tag_a_dropdown.get('class', []))

            if not active_tag_part and is_active_tag_dropdown:
                active_tag_part = tag_part_from_dropdown

            all_tags_dropdown_summary.append(TagSummary(
                tag_part=tag_part_from_dropdown,
                name_full_tag=make_full_tag_name(namespace, model_base_name, tag_part_from_dropdown),
                size_str=size_str_from_dropdown,
                is_active=(active_tag_part == tag_part_from_dropdown)
            ))

    if not active_tag_part and all_tags_dropdown_summary:
         active_tag_part = all_tags_dropdown_summary[0].tag_part
         all_tags_dropdown_summary[0].is_active = True
    elif active_tag_part:
        found_active = False
        for ts in all_tags_dropdown_summary:
            ts.is_active = (ts.tag_part == active_tag_part)
            if ts.is_active:
                found_active = True

    active_tag_full_name = make_full_tag_name(namespace, model_base_name, active_tag_part) if active_tag_part else None


    all_tags_page_link = soup.select_one('a[x-test-tags-link]')
    all_tags_page_url_str = f"{OLLAMA_COM_BASE_URL}/{namespace}/{model_base_name}/tags" # Default construction
    if all_tags_page_link and all_tags_page_link.has_attr('href'):
        all_tags_page_url_str = urljoin(OLLAMA_COM_BASE_URL, all_tags_page_link['href'])
    
    total_tags_count_from_link = 0
    if all_tags_page_link:
        count_match = re.search(r'(\d+)\s+Tags', all_tags_page_link.get_text())
        if count_match: total_tags_count_from_link = int(count_match.group(1))

    return {
        "name_full_model": make_full_model_name(namespace, model_base_name),
        "namespace": namespace,
        "model_base_name": model_base_name,
        "active_tag_part": active_tag_part,
        "active_tag_full_name": active_tag_full_name,
        "source_url": page_url, "summary": summary, "pull_count_str": pull_count_str,
        "pull_count": pull_count, "last_updated_str": last_updated_str,
        "last_updated_iso": last_updated_iso.isoformat(), "capabilities": capabilities,
        "sizes": sizes, "readme_content": readme_content, "tag_command": tag_command,
        "tag_files_summary": tag_files_summary,
        "all_tags_dropdown_summary": all_tags_dropdown_summary,
        "all_tags_page_url": all_tags_page_url_str,
        "total_tags_count_from_link": total_tags_count_from_link
    }


def parse_all_tags_page_html(html_content: str, page_url: str, model_namespace: str, model_base_name_in: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_content, 'html.parser')
    tags_list = []

    list_items = soup.select('ul > li.group.p-3')
    for item_li in list_items:
        tag_anchor = item_li.select_one('a.hover\\:underline')
        if not tag_anchor or not tag_anchor.has_attr('href'): continue

        full_tag_name_text_raw = tag_anchor.get_text(strip=True)
        expected_full_model_name = make_full_model_name(model_namespace, model_base_name_in).lower()
        if ':' in full_tag_name_text_raw:
            parts = full_tag_name_text_raw.split(':',1) # Split only on first colon
            if len(parts) == 2 and parts[0].lower() == expected_full_model_name:
                 tag_part = parts[1]
            else:
                 continue
        else:
            if full_tag_name_text_raw.lower() == expected_full_model_name:
                 tag_part = "latest"
            else:
                 continue


        source_url = urljoin(OLLAMA_COM_BASE_URL, tag_anchor['href'])

        digest_span = item_li.select_one('div.font-mono.text-\\[13px\\]')
        digest = digest_span.get_text(strip=True) if digest_span else "unknown-digest"

        size_str, context_window_str, input_type, modified_str = "N/A", None, None, "N/A"
        modified_iso = datetime.now(timezone.utc)

        details_div = item_li.select_one('div.hidden.md\\:grid')
        if details_div:
            cols = details_div.select('div.grid.grid-cols-12 > div')
            if len(cols) > 1: size_str = cols[1].get_text(strip=True)
            if len(cols) > 2: context_window_str = cols[2].get_text(strip=True) if cols[2].get_text(strip=True) != '-' else None
            if len(cols) > 3: input_type = cols[3].get_text(strip=True) if cols[3].get_text(strip=True) != '-' else None
            if len(cols) > 4:
                modified_str = cols[4].get_text(strip=True)
                modified_span_title = cols[4].select_one('span[title]')
                if modified_span_title and modified_span_title.has_attr('title'):
                     try:
                          modified_iso = parse_ollama_absolute_date_str(modified_span_title['title'])
                     except ValueError:
                          modified_iso = parse_relative_date_to_datetime(modified_str)
                else:
                     modified_iso = parse_relative_date_to_datetime(modified_str)

        else:
            mobile_details_span = item_li.select_one('a.md\\:hidden span:not([class*="group-hover:underline"])')
            if mobile_details_span:
                all_texts = [s.strip() for s in mobile_details_span.find_all(string=True, recursive=True) if s.strip()]
                full_text = " ".join(all_texts)
                parts = [p.strip() for p in full_text.split('•')]

                if len(parts) > 0:
                    digest_match = re.search(r"[0-9a-f]{7,}", parts[0]) # Shorter match for mobile digest
                    if digest_match: digest = digest_match.group(0)

                for part_idx, part_text in enumerate(parts):
                    part_lower = part_text.lower()
                    if 'gb' in part_lower or 'mb' in part_lower or 'kb' in part_lower: size_str = part_text
                    elif 'context' in part_lower: context_window_str = part_text.replace('context','').strip()
                    elif 'input' in part_lower: input_type = part_text.replace('input','').strip()
                    elif any(kw in part_lower for kw in ['ago', 'yesterday', 'now', 'updated', 'modified']):
                         modified_str = part_text
                         modified_iso = parse_relative_date_to_datetime(modified_str)


        is_default_badge = item_li.select_one('span.text-blue-600:contains("Default")')
        is_default = bool(is_default_badge)

        tags_list.append(TagDetailItem(
            name_full_tag=make_full_tag_name(model_namespace, model_base_name_in, tag_part),
            tag_part=tag_part, source_url=source_url, digest=digest,
            size_str=size_str, size_bytes=parse_size_str_to_bytes(size_str),
            context_window_str=context_window_str if context_window_str and context_window_str != '-' else None,
            input_type=input_type if input_type and input_type != '-' else None,
            modified_str=modified_str, modified_iso=modified_iso, is_default=is_default
        ))

    return {
        "name_full_model": make_full_model_name(model_namespace, model_base_name_in),
        "namespace": model_namespace,
        "model_base_name": model_base_name_in,
        "tags_page_url": page_url,
        "tags": tags_list
    }


def parse_blob_content_page_html(html_content: str) -> Optional[str]:
    """Parses a blob content page (e.g., for params, template) to extract text from <pre>."""
    soup = BeautifulSoup(html_content, 'html.parser')
    pre_tag = soup.select_one('pre')
    if pre_tag:
        return pre_tag.get_text()
    return None

def parse_gguf_metadata_from_snippet(snippet: str) -> Optional[GGUFMetadata]:
    if not snippet or not isinstance(snippet, str): return None

    metadata = {}
    parts = re.split(r'[·, ]+', snippet.lower().strip())

    for part in parts:
        if part.startswith("arch:"):
            metadata["arch"] = part.replace("arch:", "").strip()
        elif part.startswith("parameters:"):
            metadata["parameters"] = part.replace("parameters:", "").strip().upper()
        elif part.startswith("quantization:"):
            metadata["quantization"] = part.replace("quantization:", "").strip().upper()
        elif re.match(r"^[^:]+$", part):
             if re.match(r"^\w+$", part) and "arch" not in metadata and not any(c.isdigit() for c in part): metadata["arch"] = part
             elif re.match(r"^\d+(\.\d+)?[a-z]+$", part) and "parameters" not in metadata: metadata["parameters"] = part.upper()
             elif (re.match(r"^[fq]\d+(_\d+|[a-z_]+)?$", part) or part in ["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0"]) and "quantization" not in metadata: metadata["quantization"] = part.upper()


    return GGUFMetadata(**metadata) if metadata else None


# --- API Endpoints (Reordered for FastAPI matching) ---



app = FastAPI()

# Store the last N ping durations (in seconds)
ping_durations = deque(maxlen=100)

# Store the total bytes sent (simulated bandwidth usage)
total_bytes_sent = 0

@app.get("/ping")
async def ping(request: Request):
    global total_bytes_sent

    # Simulated pong response
    response_data = {"message": "pong"}
    response = JSONResponse(content=response_data)

    # Approximate response size in bytes
    response_body = response.body
    response_size = len(response_body)
    total_bytes_sent += response_size

    metrics = {
        "status": "pong",
        "total_pings": len(ping_durations),
        "total_bandwidth_sent_kb": round(total_bytes_sent / 1024, 2),
        "last_response_size_bytes": response_size
    }

    return JSONResponse(content=metrics)


@app.get("/", include_in_schema=False)
async def read_index():
    script_dir = os.path.dirname(__file__)
    html_file_path = os.path.join(script_dir, "static", FILETEMPTEXT)
    if not os.path.exists(html_file_path):
        return HTMLResponse(content="<h1>Ollama Library API</h1><p>index.html not found. See <a href='/docs'>API Documentation</a>.</p>", status_code=404)
    return FileResponse(path=html_file_path, media_type="text/html")


@app.get("/search", response_model=SearchResponse, summary="Search Models")
async def search_models(
    q: str = Query(..., description="The search query."),
    o: Optional[str] = Query("popular", description="Sort order: 'popular' or 'newest'."),
    c: Optional[str] = Query(None, description="Comma-separated list of capabilities to filter by.")
):
    if o not in ["popular", "newest"]:
        raise HTTPException(status_code=400, detail="Invalid sort order 'o'. Must be 'popular' or 'newest'.")

    params = {"q": q, "o": o}
    if c:
        params["c"] = c.lower()

    search_url = f"{OLLAMA_COM_BASE_URL}/search"
    try:
        response = cached_session.get(search_url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch search results from ollama.com: {e}")

    cache_info = get_cache_info_from_response(response)
    parsed_results_data = parse_list_or_search_page_html(response.text, base_url=OLLAMA_COM_BASE_URL)

    final_results = []
    capabilities_list_for_filterinfo = None
    if c:
        requested_caps = set(cap.strip().lower() for cap in c.split(','))
        capabilities_list_for_filterinfo = sorted(list(requested_caps))
        for model_data in parsed_results_data:
            model_caps_set = set(model_data.get("capabilities", []))
            if requested_caps.issubset(model_caps_set):
                final_results.append(ModelResultItem(**model_data))
    else:
        final_results = [ModelResultItem(**model_data) for model_data in parsed_results_data]

    return SearchResponse(
        query=q, sort_order=o,
        filters=FilterInfo(capabilities=capabilities_list_for_filterinfo) if capabilities_list_for_filterinfo else None,
        results=final_results,
        **cache_info
    )

@app.get("/{namespace}/{model_base_name}:{tag_part}/blobs/{blob_identifier}", response_model=BlobDetailsResponse, summary="Get Blob Information")
async def get_blob_information(
    namespace: str = FastApiPath(..., description="Model namespace."),
    model_base_name: str = FastApiPath(..., description="Base name of the model."),
    tag_part: str = FastApiPath(..., description="The tag part (e.g., '8b')."),
    blob_identifier: str = FastApiPath(..., description="Blob filename (e.g. 'model', 'params', 'template') or digest.")
):
    norm_ns = namespace.lower()
    norm_mbn = model_base_name.lower()
    norm_tp = tag_part.lower()
    norm_bi = blob_identifier.lower()

    tag_page_url = f"{OLLAMA_COM_BASE_URL}/{norm_ns}/{norm_mbn}:{norm_tp}"
    try:
        tag_page_response = cached_session.get(tag_page_url)
        if tag_page_response.status_code == 404:
             if norm_ns != 'library':
                 library_tag_url = f"{OLLAMA_COM_BASE_URL}/library/{norm_mbn}:{norm_tp}"
                 tag_page_response = cached_session.get(library_tag_url)
                 if tag_page_response.status_code == 404:
                     raise HTTPException(status_code=404, detail=f"Tag page for '{make_full_tag_name(namespace, model_base_name, tag_part)}' not found.")
                 norm_ns = 'library'
                 tag_page_url = library_tag_url
             else:
                raise HTTPException(status_code=404, detail=f"Tag page for '{make_full_tag_name(namespace, model_base_name, tag_part)}' not found.")
        tag_page_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch tag page: {e}")

    parsed_tag_page = parse_model_page_html(tag_page_response.text, tag_page_url)

    found_file_summary: Optional[FileSummary] = None
    for fs in parsed_tag_page.get("tag_files_summary", []):
        if fs.name.lower() == norm_bi or (fs.digest and fs.digest.lower().startswith(norm_bi)):
            found_file_summary = fs
            break

    if not found_file_summary:
        raise HTTPException(status_code=404, detail=f"Blob identifier '{blob_identifier}' not found for tag '{make_full_tag_name(norm_ns, norm_mbn, norm_tp)}'.")

    text_content = None
    parsed_json_content = None
    gguf_metadata_snippet = None
    parsed_gguf_metadata = None
    blob_detail_url = found_file_summary.blob_url
    blob_cache_info = get_cache_info_from_response(tag_page_response)
    text_blob_names = ["params", "template", "license", "modelfile"]

    if found_file_summary.name in text_blob_names:
        try:
            blob_content_response = cached_session.get(str(blob_detail_url))
            blob_content_response.raise_for_status()
            blob_cache_info = get_cache_info_from_response(blob_content_response)
            content_type = blob_content_response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                 text_content = parse_blob_content_page_html(blob_content_response.text)
            else:
                 text_content = blob_content_response.text
            if text_content and found_file_summary.name == "params":
                try: parsed_json_content = json.loads(text_content)
                except json.JSONDecodeError: pass
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch blob content from {blob_detail_url}: {e}. Using snippet.")
            text_content = found_file_summary.snippet
    elif found_file_summary.name == "model":
        gguf_metadata_snippet = found_file_summary.snippet
        if gguf_metadata_snippet:
            parsed_gguf_metadata = parse_gguf_metadata_from_snippet(gguf_metadata_snippet)
    else:
        text_content = found_file_summary.snippet

    listing_updated_iso_val = None
    if found_file_summary.updated_str: # updated_str is from the file listing section overall
         listing_updated_iso_val = parse_relative_date_to_datetime(found_file_summary.updated_str)

    return BlobDetailsResponse(
        name_full_tag=make_full_tag_name(norm_ns, norm_mbn, norm_tp),
        canonical_name=found_file_summary.name,
        source_url=blob_detail_url,
        digest=found_file_summary.digest or "unknown-digest",
        size_str=found_file_summary.size_str,
        text_content=text_content,
        parsed_json_content=parsed_json_content,
        gguf_metadata_snippet=gguf_metadata_snippet,
        parsed_gguf_metadata=parsed_gguf_metadata,
        listing_updated_str=found_file_summary.updated_str,
        listing_updated_iso=listing_updated_iso_val,
        **blob_cache_info
    )

@app.get("/{namespace}/{model_base_name}:{tag_part}", response_model=ModelPageResponse, summary="Get Specific Tag Details")
async def get_specific_tag_details(
    namespace: str = FastApiPath(..., description="Model namespace."),
    model_base_name: str = FastApiPath(..., description="Base name of the model."),
    tag_part: str = FastApiPath(..., description="The tag part (e.g., 'latest', '8b').")
):
    norm_ns = namespace.lower()
    norm_mbn = model_base_name.lower()
    norm_tp = tag_part.lower()
    tag_page_url = f"{OLLAMA_COM_BASE_URL}/{norm_ns}/{norm_mbn}:{norm_tp}"

    try:
        response = cached_session.get(tag_page_url)
        if response.status_code == 404:
            if norm_ns != 'library':
                 library_tag_url = f"{OLLAMA_COM_BASE_URL}/library/{norm_mbn}:{norm_tp}"
                 response = cached_session.get(library_tag_url)
                 if response.status_code == 404:
                     raise HTTPException(status_code=404, detail=f"Tag page '{namespace}/{model_base_name}:{tag_part}' not found.")
                 norm_ns = 'library'
                 tag_page_url = library_tag_url
            else:
                 raise HTTPException(status_code=404, detail=f"Tag page '{namespace}/{model_base_name}:{tag_part}' not found.")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch specific tag page: {e}")

    cache_info = get_cache_info_from_response(response)
    parsed_data = parse_model_page_html(response.text, tag_page_url)
    parsed_data['active_tag_part'] = norm_tp
    parsed_data['active_tag_full_name'] = make_full_tag_name(norm_ns, norm_mbn, norm_tp)
    for ts in parsed_data.get("all_tags_dropdown_summary", []):
        ts.is_active = (ts.tag_part == norm_tp)

    return ModelPageResponse(**parsed_data, **cache_info)


@app.get("/{namespace}/{model_base_name}/tags", response_model=AllTagsResponse, summary="List All Tags for a Model")
async def list_all_tags(
    namespace: str = FastApiPath(..., description="Model namespace."),
    model_base_name: str = FastApiPath(..., description="Base name of the model.")
):
    norm_ns = namespace.lower()
    norm_mbn = model_base_name.lower()
    tags_page_url = f"{OLLAMA_COM_BASE_URL}/{norm_ns}/{norm_mbn}/tags"
    try:
        response = cached_session.get(tags_page_url)
        if response.status_code == 404:
             if norm_ns != 'library':
                 library_tags_url = f"{OLLAMA_COM_BASE_URL}/library/{norm_mbn}/tags"
                 response = cached_session.get(library_tags_url)
                 if response.status_code == 404:
                     raise HTTPException(status_code=404, detail=f"Tags page for model '{namespace}/{model_base_name}' not found.")
                 norm_ns = 'library'
                 tags_page_url = library_tags_url
             else:
                 raise HTTPException(status_code=404, detail=f"Tags page for model '{namespace}/{model_base_name}' not found.")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch tags page: {e}")

    cache_info = get_cache_info_from_response(response)
    parsed_data = parse_all_tags_page_html(response.text, tags_page_url, norm_ns, norm_mbn)

    return AllTagsResponse(**parsed_data, **cache_info)


@app.get("/{namespace}/{model_base_name}", response_model=ModelPageResponse, summary="Get Model Details")
async def get_model_details(
    namespace: str = FastApiPath(..., description="Model namespace."),
    model_base_name: str = FastApiPath(..., description="Base name of the model.")
):
    page_url = f"{OLLAMA_COM_BASE_URL}/{namespace.lower()}/{model_base_name.lower()}"
    try:
        response = cached_session.get(page_url)
        if response.status_code == 404:
             # Try with 'library' namespace if original was not 'library'
            if namespace.lower() != 'library':
                lib_page_url = f"{OLLAMA_COM_BASE_URL}/library/{model_base_name.lower()}"
                response = cached_session.get(lib_page_url)
                if response.status_code == 404:
                    raise HTTPException(status_code=404, detail=f"Model '{namespace}/{model_base_name}' (and as 'library/{model_base_name}') not found.")
                page_url = lib_page_url # Use library URL if found
            else: # Original was 'library' and not found
                raise HTTPException(status_code=404, detail=f"Model '{namespace}/{model_base_name}' not found.")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch model details: {e}")

    cache_info = get_cache_info_from_response(response)
    parsed_data = parse_model_page_html(response.text, page_url)
    return ModelPageResponse(**parsed_data, **cache_info)


@app.get("/{namespace}", response_model=ModelListByNamespaceResponse, summary="List Models by Namespace")
async def list_models_by_namespace(
    namespace: str = FastApiPath(..., description="The model namespace (e.g., 'library', 'username')."),
    o: Optional[str] = Query('popular', description="Sort order: 'newest' or 'popular'."),
    c: Optional[str] = Query(None, description="Comma-separated list of capabilities to filter by.")
):
    if o not in ["newest", "popular"]:
        raise HTTPException(status_code=400, detail="Sort order 'o' must be 'newest' or 'popular'.")

    norm_namespace = namespace.lower()
    
    # Determine the actual path on ollama.com
    if norm_namespace == "library":
        ollama_com_path = "/library"
    else:
        ollama_com_path = f"/{norm_namespace}"

    target_fetch_url = f"{OLLAMA_COM_BASE_URL}{ollama_com_path}"
    
    params = {"sort": o} # ollama.com uses 'sort' for both /library and /{user} pages
    if c:
        params["c"] = c.lower()

    try:
        response = cached_session.get(target_fetch_url, params=params)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found on ollama.com.")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch models for namespace '{namespace}': {e}")

    cache_info = get_cache_info_from_response(response)
    # The parse_list_or_search_page_html should work for both /library and /user_name pages
    # as the HTML structure for model listings is similar.
    parsed_results_data = parse_list_or_search_page_html(response.text, base_url=OLLAMA_COM_BASE_URL)

    final_results = []
    capabilities_list_for_filterinfo = None
    if c:
        requested_caps = set(cap.strip().lower() for cap in c.split(','))
        capabilities_list_for_filterinfo = sorted(list(requested_caps))
        for model_data in parsed_results_data:
            model_caps_set = set(model_data.get("capabilities", []))
            if requested_caps.issubset(model_caps_set):
                final_results.append(ModelResultItem(**model_data))
    else:
        final_results = [ModelResultItem(**model_data) for model_data in parsed_results_data]

    return ModelListByNamespaceResponse(
        queried_namespace=norm_namespace,
        sort_order=o,
        filters=FilterInfo(capabilities=capabilities_list_for_filterinfo) if capabilities_list_for_filterinfo else None,
        results=final_results,
        **cache_info
    )


if not format.startswith('.'):
    format = f'.{format}'
if not Static_Website and os.path.exists('static/index.html'):
    os.remove('static/index.html')

if Static_Website == False:
    filename = f'{FILETEMPTEXT}'
# --- Main execution ---
if __name__ == "__main__":
    print("Starting Ollama Library API (Live Fetch) server...")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        print(f"Created static directory: {static_dir}")

    index_html_path = os.path.join(static_dir, filename)
    if not os.path.exists(index_html_path):
        dummy_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama API Proxy</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-third: #111927;
            --accent-primary: #38bdf8;
            --accent-secondary: #7dd3fc;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --gradient: linear-gradient(135deg, #38bdf8 0%, #7dd3fc 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .header {
            text-align: center;
            padding: 4rem 0;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
            margin-bottom: 3rem;
            transition: 0.7s ease-in-out;
        }

        .title {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        .nav-links {
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            margin-bottom: 2rem;
        }

        .nav-link {
            color: var(--accent-primary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: 0.6s ease-in-out;
        }

        .nav-link:hover {
            color: var(--accent-secondary);
            transition: 0.3s ease-in-out;
            background: rgba(56, 189, 248, 0.1);
        }

        .section {
            margin-bottom: 3rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .section-title {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .endpoint-grid {
            display: grid;
            gap: 1.5rem;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        }

        .endpoint-card {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            transition: transform 0.2s ease;
        }

        .endpoint-card:hover {
            transform: translateY(-3px);
        }

        .endpoint-title {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-primary);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .endpoint-description {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .code-snippet {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            position: relative;
            margin: 1rem 0;
        }

        .copy-button {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: var(--bg-secondary);
            border: none;
            color: var(--text-secondary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .copy-button:hover {
            color: var(--accent-primary);
            background: var(--bg-primary);
        }

        .footer {
            text-align: center;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color);
            margin-top: 3rem;
            color: var(--text-secondary);
        }

        .social-links {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 1rem;
        }

        .social-link {
            color: var(--text-secondary);
            transition: color 0.3s ease;
        }

        .social-link:hover {
            color: var(--accent-primary);
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .title {
                font-size: 2rem;
            }
            
            .endpoint-grid {
                grid-template-columns: 1fr;
            }
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
            background: rgba(56, 189, 248, 0.1);
            color: var(--accent-primary);
            margin-left: 0.5rem;
        }

        .animate-fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

            .apps-grid {
        display: grid;
        gap: 2rem;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }

    .app-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .app-card:hover {
        border-color: var(--accent-primary);
    }

    .app-icon {
        width: 80px;
        height: 80px;
        object-fit: contain;
        margin: 0 auto 1rem;
        border-radius: 16px;
        filter: grayscale(1);
        transition: filter 0.3s ease;
    }

    .app-card:hover .app-icon {
        filter: grayscale(0);
    }

    .app-name {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .app-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .app-version {
        font-size: 0.75rem;
        color: var(--accent-primary);
        background: rgba(56, 189, 248, 0.1);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
    }

            .apps-grid {
            display: grid;
            gap: 2rem;
            grid-template-columns: repeat(4, 1fr);
            max-height: 400px;
            overflow-y: auto;
        }

        @media (max-width: 1200px) {
            .apps-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (max-width: 768px) {
            .apps-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 480px) {
            .apps-grid {
                grid-template-columns: 1fr;
            }
        }

        .submit-form {
            background: var(--bg-primary);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 2rem;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        .form-input {
            width: 100%;
            padding: 0.8rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 6px;
            margin-top: 0.5rem;
        }

        .submit-button {
            background: var(--bg-primary);
            color: var(--text-primary);
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            transition: opacity 0.3s ease;
            width: 100%;
        }

        .submit-button:hover {
            opacity: 0.9;
        }

        .status-message {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 6px;
            display: none;
        }

        .success {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid var(--accent-primary);
        }

        .error {
            background: rgba(248, 56, 56, 0.1);
            border: 1px solid #f87171;
        }

            .modal {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 23, 42, 0.95);
        z-index: 1000;
        justify-content: center;
        align-items: center;
        animation: fadeIn 0.3s ease-out;
    }

    .modal-content {
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 12px;
        width: 90%;
        max-width: 500px;
        position: relative;
    }

    .close-modal {
        position: absolute;
        top: 1rem;
        right: 1rem;
        color: var(--text-secondary);
        font-size: 1.5rem;
        cursor: pointer;
        transition: color 0.3s ease;
    }

    .close-modal:hover {
        color: var(--accent-primary);
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
</head>
<body>
    <header class="header animate-fade-in">
        <div class="container">
            <h1 class="title">Ollama API Proxy</h1>
            <p class="subtitle">An API that fetches, parses, and caches data from ollama.com.</p>
            <nav class="nav-links">
                <a href="/docs" class="nav-link"><i class="fas fa-code"></i> API Docs</a>
                <a href="/redoc" class="nav-link"><i class="fas fa-book-open"></i> ReDoc</a>
                <a href="https://ollama.com" target="_blank" class="nav-link"><i class="fas fa-external-link-alt"></i> Ollama.com</a>
            </nav>
        </div>
    </header>

    <main class="container">
        <section class="section animate-fade-in">
            <h2 class="section-title"><i class="fas fa-rocket"></i> Getting Started</h2>
            <div class="code-snippet">
                <button class="copy-button" onclick="navigator.clipboard.writeText('http://localhost:5115/docs')">
                    <i class="far fa-copy"></i>
                </button>
                # Explore the API documentation
                $ open http://localhost:5115/docs
            </div>
        </section>

        <section class="section animate-fade-in">
            <h2 class="section-title"><i class="fas fa-plug"></i> Example Endpoints</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card">
                    <div class="endpoint-title">GET /library?o=popular <span class="badge">Default</span></div>
                    <p class="endpoint-description">Get popular models from official library</p>
                </div>
                
                <div class="endpoint-card">
                    <div class="endpoint-title">GET /jmorganca/llama3 <span class="badge">User Model</span></div>
                    <p class="endpoint-description">Get details for specific user model</p>
                </div>

                <div class="endpoint-card">
                    <div class="endpoint-title">GET /search?q=mistral <span class="badge">Search</span></div>
                    <p class="endpoint-description">Global model search functionality</p>
                </div>

                <div class="endpoint-card">
                    <div class="endpoint-title">GET /.../blobs/model <span class="badge">Blobs</span></div>
                    <p class="endpoint-description">Access raw model artifacts</p>
                </div>
            </div>
        </section>
        

        <section class="section animate-fade-in">
            <h2 class="section-title"><i class="fas fa-microchip"></i> System Status</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card">
                    <div class="endpoint-title">Cache Status <span class="badge">Live</span></div>
                    <p class="endpoint-description">6-hour intelligent caching</p>
                </div>
                <div class="endpoint-card">
                    <div class="endpoint-title">Uptime <span class="badge">99.9%</span></div>
                    <p class="endpoint-description">High availability service</p>
                </div>
            </div>
        </section>

        <section class="section animate-fade-in">
            <h2 class="section-title"><i class="fas fa-rocket"></i> Powered Apps</h2>
            <div class="apps-grid" id="apps-container"></div>
            <button class="submit-button" id="openModal" style="margin-top: 1.5rem;">
                <i class="fas fa-plus"></i> Submit Your App
            </button>
        </section>
    </main>

<div id="submitModal" class="modal">
    <div class="modal-content">
        <span class="close-modal">&times;</span>
        <h3 class="section-title"><i class="fas fa-plus-circle"></i> Submit Your App</h3>
        <form id="appSubmitForm" onsubmit="return submitApp(event)">
            <div class="form-group">
                <label>App Name</label>
                <input type="text" class="form-input" id="appName" required>
            </div>
            <div class="form-group">
                <label>Website URL</label>
                <input type="url" class="form-input" id="websiteUrl" required>
            </div>
            <button type="submit" class="submit-button">Submit Application</button>
        </form>
        <div id="statusMessage" class="status-message"></div>
    </div>
</div>

    <footer class="footer animate-fade-in">
        <div class="container">
            <div class="social-links" style="margin-bottom: 20px;">
                <a href="https://github.com/Houloude9IOfficial/OllamaSearchAPI" class="social-link" target="_blank">
                    <i class="fab fa-github"></i>
                </a>
                <a href="https://houloude9.is-a.dev" class="social-link" target="_blank">
                    <i class="fa fa-globe"></i>
                </a>
            </div>
            <span id="version" class="badge"></span>
            <p>
              Developed by 
              <a href="https://discord.com/users/575254127748317194" target="_blank" rel="noopener noreferrer"
                 style="text-decoration: none; font-weight: bold; color: inherit;"
                 onmouseover="this.style.textDecoration='underline'" 
                 onmouseout="this.style.textDecoration='none'">
                Blood Shot
              </a>
            </p>
            <p>
              Maintained by 
              <a href="https://discord.com/users/947432701160480828" target="_blank" rel="noopener noreferrer"
                 style="text-decoration: none; font-weight: bold; color: inherit;"
                 onmouseover="this.style.textDecoration='underline'" 
                 onmouseout="this.style.textDecoration='none'">
                Houloude9
              </a>
            </p>
            <p>
              Powered by 
              <a href="https://render.com" target="_blank" rel="noopener noreferrer"
                 style="text-decoration: none; font-weight: bold; color: inherit;"
                 onmouseover="this.style.textDecoration='underline'" 
                 onmouseout="this.style.textDecoration='none'">
                Render.com
              </a>
            </p>

        </div>
    </footer>

    <script>
        function setversion(version) {
            if(!String(String(version).toLowerCase()).startsWith('v')) {
                version = `v${version}`
            }
            document.getElementById('version').textContent = version
        }
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });

        document.querySelectorAll('.copy-button').forEach(button => {
            button.addEventListener('click', function() {
                const snippet = this.parentElement.textContent.replace('Copy', '').trim();
                navigator.clipboard.writeText(snippet);
                
                const originalHTML = this.innerHTML;
                this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                
                setTimeout(() => {
                    this.innerHTML = originalHTML;
                }, 2000);
            });
        });
            const modal = document.getElementById('submitModal');
    const openBtn = document.getElementById('openModal');
    const closeSpan = document.querySelector('.close-modal');

    openBtn.onclick = () => modal.style.display = 'flex';
    closeSpan.onclick = () => modal.style.display = 'none';

    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    }
        
        async function submitApp(event) {
            event.preventDefault();
            const name = document.getElementById('appName').value;
            const website = document.getElementById('websiteUrl').value;
            const statusMessage = document.getElementById('statusMessage');

            try {
                new URL(website);
            } catch {
                statusMessage.textContent = "Please enter a valid URL";
                statusMessage.className = "status-message error";
                statusMessage.style.display = 'block';
                return;
            }

            try {
                const response = await fetch('https://nextuiserver.htdevs.workers.dev/ollamasearchapi/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name, website }),
                });

                const data = await response.json();
                
        if (response.ok) {
            statusMessage.textContent = "App submitted successfully!";
            statusMessage.className = "status-message success";
            document.getElementById('appSubmitForm').reset();
            setTimeout(() => {
                modal.style.display = 'none';
                statusMessage.style.display = 'none';
            }, 2000);
        } else {
                    statusMessage.textContent = data.error || "Submission failed";
                    statusMessage.className = "status-message error";
                }
            } catch (error) {
                statusMessage.textContent = "Network error - please try again";
                statusMessage.className = "status-message error";
            }
            
            statusMessage.style.display = 'block';
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }

        async function loadPoweredApps() {
            try {
                const response = await fetch('https://nextuiserver.htdevs.workers.dev/ollamasearchapi/getapps');
                const data = await response.json();
                const container = document.getElementById('apps-container');
                
                container.innerHTML = '';
                
                data.apps.forEach(app => {
                    const card = document.createElement('div');
                    card.className = 'app-card';
                    card.onclick = () => window.open(app.url, '_blank');
                    const fullversion = app.version ? `<div class="app-version">${app.version}</div>` : '';
                    
                    card.innerHTML = `
                        <img src="${app.icon}" class="app-icon" alt="${app.name}">
                        <div class="app-name">${app.name}</div>
                        <div class="app-description">${app.description}</div>
                        ${fullversion}
                    `;
                    
                    container.appendChild(card);
                });
            } catch (error) {
                console.error('Error loading powered apps:', error);
            }
        }

    window.addEventListener('DOMContentLoaded', loadPoweredApps);

        setversion('VERSION_BEING_REPLACED')
    </script>
</body>
</html>
"""
    # Replace placeholder before writing to file
        modified_html_content = dummy_html_content.replace("VERSION_BEING_REPLACED", CODE_VERSION)
        
        # Write the modified content
        with open(index_html_path, "w") as f:
            f.write(modified_html_content)
        
        print(f"Created dummy index.html at: {index_html_path}")

    print("OpenAPI docs available at http://localhost:5115/docs")
    print("Landing page available at http://localhost:5115/")
    print("\nExample test URLs (fetches live data from ollama.com):")
    print("  Models in 'library' namespace (popular): http://localhost:5115/library?o=popular")
    print("  Models by user 'jmorganca' (popular): http://localhost:5115/jmorganca?o=popular")
    print("  Models in 'library' (newest, filter vision): http://localhost:5115/library?o=newest&c=vision")
    print("  Search 'qwen': http://localhost:5115/search?q=qwen&o=popular")
    print("  Model details (library/qwen2): http://localhost:5115/library/qwen2")
    print("  Model details (jmorganca/codellama): http://localhost:5115/jmorganca/codellama")
    print("  Model specific tag (library/qwen2:7b): http://localhost:5115/library/qwen2:7b")
    print("  All tags for (library/qwen2): http://localhost:5115/library/qwen2/tags")
    print("  Blob info (library/qwen2:7b, model): http://localhost:5115/library/qwen2:7b/blobs/model")
    print("  Blob info (library/qwen2:7b, params): http://localhost:5115/library/qwen2:7b/blobs/params")

    uvicorn.run(app, host="0.0.0.0", port=5115)