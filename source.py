import os

from constant import SOURCE_FILE, WATCH_CLIENT_FILE

def get_source() -> dict[str, str]:
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as file:
            result: dict[str, str]= {}
            for line in file:
                item = line.strip().split(": ")
                if len(item) == 2:
                    result[item[0]] = item[1]
            return result
    except FileNotFoundError:
        return {}

def put_source(key: str, src: str):
    #source文件不存在
    if not os.path.exists(SOURCE_FILE):
        with open(SOURCE_FILE, 'w') as f:
            _ = f.write(f"{key}: {src}")
    else:
        sources = get_source()
        sources[key] = src
        
        items: list[str]= []
        for key in sources.keys():
            items.append(f"{key}: {sources[key]}")
        if len(items) == 0:
            return
        with open(SOURCE_FILE, 'w') as f:
            _ = f.write("\n".join(items))

def del_source(key: str):
    #source文件不存在
    if not os.path.exists(SOURCE_FILE):
        return

    #source文件存在，删除文件记录 & 删除视频流
    sources = get_source()
    if key in sources.keys():
        del sources[key]
        items: list[str]= []
        for key in sources.keys():
            items.append(f"{key}: {sources[key]}")
        if len(items) == 0:
            return
        with open(SOURCE_FILE, 'w') as f:
            _ = f.write("\n".join(items))


def get_watch_client() -> list[str]:
    try:
        with open(WATCH_CLIENT_FILE, 'r', encoding='utf-8') as file:
            result: list[str]= []
            for line in file:
                result.append(line.strip())
            return result
    except FileNotFoundError:
        return []