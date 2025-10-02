import html
import json
from typing import List, Dict, Any, Optional

from appdirs import AppDirs


def markup(text: str, style: str) -> str:
    ansi_codes = {"bold": "\033[1m", "red": "\033[31m", "green": "\033[32m",
                  "cyan": "\033[36m", "magenta": "\033[35m"}
    return ansi_codes[style] + text + "\033[0m"


def pre_format(text: str) -> str:
    escaped_text = html.escape(text)
    return f"<pre>{escaped_text}</pre>"


def split_message(message: str, max_length: int) -> List[str]:
    """Split large message into smaller messages each smaller than the max_length."""
    ms = []
    while len(message) > max_length:
        ms.append(message[:max_length])
        message = message[max_length:]
    ms.append(message)
    return ms


def get_config_path():
    return AppDirs("telegram-send").user_config_dir + ".conf"


def process_json_message(json_data: str) -> Dict[str, Any]:
    """Process message from JSON format."""
    try:
        data = json.loads(json_data)
        return {
            'text': data.get('text', ''),
            'buttons': data.get('buttons', []),
            'parse_mode': data.get('parse_mode', 'HTML'),
            'chat_id': data.get('chat_id'),
            'disable_notification': data.get('disable_notification', False)
        }
    except json.JSONDecodeError:
        return {'text': json_data, 'buttons': [], 'parse_mode': 'HTML'}


def create_inline_keyboard(buttons: List[List[Dict[str, str]]]) -> Dict[str, Any]:
    """Create inline keyboard markup from button data."""
    keyboard = []
    for row in buttons:
        keyboard_row = []
        for button in row:
            keyboard_row.append({
                'text': button.get('text', ''),
                'callback_data': button.get('callback_data', ''),
                'url': button.get('url', '')
            })
        keyboard.append(keyboard_row)
    return {'inline_keyboard': keyboard}


def get_profile_config_path(profile: str) -> str:
    """Get config path for specific profile."""
    base_dir = AppDirs("telegram-send").user_config_dir
    return f"{base_dir}_{profile}.conf"


def list_profiles() -> List[str]:
    """List available bot profiles."""
    import os
    import glob
    base_dir = AppDirs("telegram-send").user_config_dir
    pattern = f"{base_dir}_*.conf"
    profiles = []
    for config_file in glob.glob(pattern):
        profile_name = os.path.basename(config_file).replace(f"{os.path.basename(base_dir)}_", "").replace(".conf", "")
        profiles.append(profile_name)
    return profiles


def manage_chat_data(chat_id: str, action: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Manage chat-specific data storage."""
    import os
    chat_data_dir = AppDirs("telegram-send").user_data_dir
    os.makedirs(chat_data_dir, exist_ok=True)
    chat_file = os.path.join(chat_data_dir, f"chat_{chat_id}.json")
    
    if action == "save" and data:
        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    elif action == "load":
        try:
            with open(chat_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    elif action == "delete":
        try:
            os.remove(chat_file)
            return {}
        except FileNotFoundError:
            return {}
    return None