import os
import json
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 从本地读取配置文件
def load_config():
    with open('config.json', 'r') as file:
        return json.load(file)

# 定义基础URL
base_url = "https://www.ebi.ac.uk/metagenomics/api/v1"

# 设置日志记录
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s %(message)s')

# 维护已访问 URL 的集合
visited_urls = set()

# 维护已访问的 MGYS 集合
visited_mgys = set()

# 获取biomes数据
def get_biomes(page=1, page_size=25):
    config = load_config()
    cookies = config['cookies']
    headers = config['headers']

    url = f"{base_url}/biomes?page={page}&page_size={page_size}"
    if url in visited_urls:
        return None
    try:
        response = requests.get(url, headers=headers, cookies=cookies)
        response.raise_for_status()
        visited_urls.add(url)
        return response.json()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

# 构建biomes树结构
def build_biome_tree(biomes):
    tree = {}
    for biome in biomes:
        parts = biome.split(":")
        node = tree
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]
    return tree

# 获取所有biomes的id
def get_all_biome_ids():
    biomes = []
    page = 1
    first_page_data = get_biomes(page=page)
    if first_page_data:
        total_pages = first_page_data['meta']['pagination']['pages']
        biomes.extend([item['id'] for item in first_page_data['data']])

        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for page in range(total_pages, 1, -1):
                futures.append(executor.submit(get_biomes, page))

            for future in as_completed(futures):
                data = future.result()
                if data:
                    biomes.extend([item['id'] for item in data['data']])

    return biomes

# 递归地访问叶子节点的samples
def process_leaf_nodes(tree, lineage=""):
    for key in sorted(tree.keys(), reverse=True):
        subtree = tree[key]
        new_lineage = f"{lineage}:{key}" if lineage else key
        if subtree:
            process_leaf_nodes(subtree, new_lineage)
        else:
            process_samples(new_lineage)

# 处理samples
def process_samples(lineage):
    page = 1
    first_page_data = fetch_samples(f"{base_url}/samples?page={page}&lineage={lineage}&page_size=25")
    if first_page_data:
        total_pages = first_page_data['meta']['pagination']['pages']

        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for page in range(total_pages, 0, -1):
                url = f"{base_url}/samples?page={page}&lineage={lineage}&page_size=25"
                futures.append(executor.submit(fetch_samples, url))

            for future in as_completed(futures):
                data = future.result()
                if data:
                    for sample in data['data']:
                        # 检查是否有未访问的 MGYS
                        if any(study['id'] not in visited_mgys for study in sample['relationships']['studies']['data']):
                            process_studies(sample['id'], lineage)

def fetch_samples(url):
    config = load_config()
    cookies = config['cookies']
    headers = config['headers']

    if url in visited_urls:
        return None
    try:
        response = requests.get(url, headers=headers, cookies=cookies)
        response.raise_for_status()
        visited_urls.add(url)
        return response.json()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

# 处理studies
def process_studies(sample_id, lineage):
    config = load_config()
    cookies = config['cookies']
    headers = config['headers']

    url = f"{base_url}/samples/{sample_id}/studies"
    if url in visited_urls:
        return
    try:
        response = requests.get(url, headers=headers, cookies=cookies)
        response.raise_for_status()
        visited_urls.add(url)
        data = response.json()
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for study in data['data']:
                if study['id'] not in visited_mgys:
                    visited_mgys.add(study['id'])
                    futures.append(executor.submit(process_downloads, study['id'], lineage))

            for future in as_completed(futures):
                future.result()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")

# 处理downloads
def process_downloads(study_id, lineage):
    config = load_config()
    cookies = config['cookies']
    headers = config['headers']

    url = f"{base_url}/studies/{study_id}/downloads"
    if url in visited_urls:
        return
    try:
        response = requests.get(url, headers=headers, cookies=cookies)
        response.raise_for_status()
        visited_urls.add(url)
        data = response.json()
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            is_download = False
            for download in data['data']:
                if (download['attributes']['group-type'] == "Taxonomic analysis SSU rRNA" and \
                    download['attributes']['description']['label'] == "Taxonomic assignments SSU") or (
                        download['attributes']['group-type'] == "Taxonomic analysis" and \
                        download['attributes']['description']['label'] == "Taxonomic assignments"
                ):
                    download_url = download['links']['self']
                    file_name = download['attributes']['alias']
                    lineage_path = lineage.replace(":", "-")
                    directory = os.path.join(os.getcwd(), lineage_path)
                    os.makedirs(directory, exist_ok=True)  # 确保目录存在
                    file_path = os.path.join(directory, file_name)
                    # 检查文件是否已经存在
                    if not os.path.exists(file_path):
                        futures.append(executor.submit(download_file, download_url, file_path))
                    else:
                        print(f"File {file_name} already exists at {file_path}, skipping download.")
                    is_download = True
                    break
            if not is_download:
                print(f"warning: {url} not download")
            for future in as_completed(futures):
                future.result()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")

# 下载文件
def download_file(url, file_path):
    config = load_config()
    cookies = config['cookies']
    headers = config['headers']

    try:
        response = requests.get(url, headers=headers, cookies=cookies)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {os.path.basename(file_path)} to {file_path}")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")

# 主函数
def main():
    biomes = get_all_biome_ids()
    biome_tree = build_biome_tree(biomes)
    process_leaf_nodes(biome_tree)

if __name__ == "__main__":
    main()
