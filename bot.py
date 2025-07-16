# =====================================================================================
# ИМПОРТ НЕОБХОДИМЫХ БИБЛИОТЕК
# =====================================================================================

# Стандартные библиотеки Python
import logging  # Для вывода информации о работе бота в консоль (логирования)
import os       # Для работы с операционной системой, в данном случае для проверки существования файла
import requests # Для отправки HTTP-запросов к API нейросети (Google Gemini)
import re       # Для работы с регулярными выражениями (используется для очистки текста от Markdown)
import sys      # Для управления системными функциями, в данном случае для завершения работы скрипта при критических ошибках
import json     # Для работы с форматом JSON, в котором мы будем хранить состояние бота (роли и историю)
import traceback # Для вывода полной информации об ошибках (стек вызовов)
import asyncio
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Библиотека для работы с Telegram Bot API
# Устанавливается через: pip install python-telegram-bot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup # Основные классы для работы с обновлениями и инлайн-кнопками
from telegram.ext import (
    Application,
    CommandHandler,      # Обработчик для команд (например, /start, /clear)
    MessageHandler,      # Обработчик для текстовых сообщений
    filters,             # Для фильтрации входящих сообщений (например, только текст)
    ContextTypes,
    CallbackQueryHandler # Обработчик для нажатий на инлайн-кнопки
)
# from telegram.constants import FileDownloadMethod # УДАЛЕНО


# =====================================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКИ
# =====================================================================================

# Эта глобальная переменная будет хранить ВСЕ состояние бота в оперативной памяти.
# При старте она загружается из файла, а при изменениях (новая роль, новая история) - сохраняется обратно в файл.
# Структура словаря:
# {
#   "chats": {
#     "CHAT_ID_1": {
#       "roles": { "USER_ID_1": "Имя1", "USER_ID_2": "Имя2", ... },
#       "history": [ {"role": "user", "content": "Имя1: Привет!"}, {"role": "assistant", "content": "Привет, Имя1!"}, ... ]
#     },
#     "CHAT_ID_2": { ... }
#   }
# }
bot_state = {'chats': {}}

# --- Настройка системы логирования ---
# Это нужно делать как можно раньше, чтобы все последующие действия (включая ошибки при загрузке)
# могли быть записаны в лог с указанием времени и уровня важности.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Формат вывода логов
    level=logging.INFO                                           # Уровень логирования (INFO и выше: WARNING, ERROR, CRITICAL)
)
# Уменьшаем количество логов от библиотеки httpx (которую использует python-telegram-bot),
# чтобы не засорять консоль технической информацией о запросах.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Создаем объект логгера для нашего скрипта.
logger = logging.getLogger(__name__)


# =====================================================================================
# ФУНКЦИИ-ПОМОЩНИКИ (Helpers)
# =====================================================================================

def load_config(filename='conf.txt'):
    """
    Читает конфигурационные переменные из текстового файла.

    Файл должен иметь формат "КЛЮЧ=ЗНАЧЕНИЕ" на каждой строке.
    Строки, начинающиеся с '#', и пустые строки игнорируются.

    Args:
        filename (str): Имя файла конфигурации.

    Returns:
        dict: Словарь с загруженными настройками.
    """
    config = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip() # Убираем пробелы по краям строки
                # Пропускаем пустые строки и строки с комментариями
                if not line or line.startswith('#'):
                    continue

                # Разделяем строку по первому знаку '='
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    config[key] = value
                else:
                    # Логгер здесь еще может быть недоступен, поэтому используем print
                    print(f"WARNING: Некорректная строка конфигурации, пропущена: {line}")
    except FileNotFoundError:
        print(f"ERROR: Файл конфигурации '{filename}' не найден. Пожалуйста, создайте его.")
        sys.exit() # Завершаем работу, так как без конфига бот бесполезен
    except Exception as e:
        print(f"ERROR: Ошибка при чтении файла конфигурации '{filename}': {e}")
        sys.exit()

    return config

def load_system_prompt(filename='prompt.txt'):
    """
    Читает системный промпт (инструкцию для нейросети) из файла.

    Args:
        filename (str): Имя файла с промптом.

    Returns:
        str: Содержимое файла в виде одной строки.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"ERROR: Файл системного промпта '{filename}' не найден. Пожалуйста, создайте его.")
        sys.exit()
    except Exception as e:
        print(f"ERROR: Ошибка при чтении файла системного промпта '{filename}': {e}")
        sys.exit()

# --- Функции для управления состоянием бота (сохранение/загрузка) ---

def load_state(filename):
    """
    Загружает состояние бота (роли и историю) из JSON файла в глобальную переменную `bot_state`.
    Если файл поврежден или отсутствует, инициализирует пустое состояние.

    Args:
        filename (str): Имя файла состояния (например, 'temp.txt').
    """
    global bot_state # Указываем, что будем модифицировать глобальную переменную
    try:
        if os.path.exists(filename): # Проверяем, существует ли файл
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Проверяем, не пустой ли файл. json.load() выдаст ошибку на пустом файле.
                if not content:
                    logger.info(f"Файл состояния {filename} пуст. Инициализация пустого состояния.")
                    bot_state = {'chats': {}}
                    return
                # Пытаемся загрузить данные из файла
                bot_state = json.loads(content)
                # Проверяем базовую структуру, чтобы убедиться, что файл не поврежден
                if 'chats' not in bot_state or not isinstance(bot_state['chats'], dict):
                     logger.warning(f"Файл состояния {filename} имеет неожиданную структуру. Инициализация пустого состояния.")
                     bot_state = {'chats': {}}
                     return
                logger.info(f"Состояние успешно загружено из {filename}.")
        else:
            logger.info(f"Файл состояния {filename} не найден. Инициализация пустого состояния.")
            bot_state = {'chats': {}} # Инициализируем пустое состояние, если файла нет
    except json.JSONDecodeError as e:
        # Эта ошибка возникает, если файл содержит невалидный JSON
        logger.error(f"Ошибка декодирования JSON из файла состояния {filename}: {e}", exc_info=True)
        logger.warning(f"Инициализация пустого состояния из-за ошибки. Пожалуйста, удалите или исправьте файл {filename}.")
        bot_state = {'chats': {}}
    except Exception as e:
        # Другие возможные ошибки при чтении файла
        logger.error(f"Ошибка при загрузке состояния из {filename}: {e}", exc_info=True)
        logger.warning(f"Инициализация пустого состояния из-за непредвиденной ошибки.")
        bot_state = {'chats': {}}

def save_state(filename):
    """
    Сохраняет текущее состояние из глобальной переменной `bot_state` в JSON файл.

    Args:
        filename (str): Имя файла состояния (например, 'temp.txt').
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Сохраняем словарь в файл в формате JSON
            # indent=4 делает файл читаемым для человека (с отступами)
            # ensure_ascii=False позволяет корректно сохранять не-ASCII символы (например, кириллицу)
            json.dump(bot_state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        # Если сохранить не удалось, бот не должен падать, но мы должны знать об ошибке.
        logger.error(f"Ошибка сохранения состояния в файл {filename}: {e}", exc_info=True)

def ensure_chat_state_exists(chat_id):
    """
    Проверяет, есть ли запись для данного чата в `bot_state`.
    Если нет, создает пустую структуру (роли и история), чтобы избежать ошибок `KeyError`.

    Args:
        chat_id (int): ID чата.
    """
    chat_id_str = str(chat_id) # JSON ключи всегда строки, поэтому работаем со строковым представлением ID
    if chat_id_str not in bot_state['chats']:
        bot_state['chats'][chat_id_str] = {'roles': {}, 'history': []}
        logger.info(f"Инициализирована структура состояния для нового чата: {chat_id}")

# --- Вспомогательная функция для очистки текста от Markdown ---
def strip_markdown(text):
    """
    Удаляет основные символы форматирования Markdown из строки.

    Args:
        text (str): Входной текст.

    Returns:
        str: Текст без Markdown.
    """
    # Удаляем жирный текст (**текст** и *текст*)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Удаляем курсив (__текст__ и _текст_)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Удаляем моноширинный текст (`текст`) и блоки кода (```текст```)
    text = re.sub(r'```(.*?)```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Удаляем цитаты (>) и маркеры списков (- или *) в начале строк
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s*', '', text, flags=re.MULTILINE)
    # Убираем лишние пробелы в строках после удаления символов
    text = "\n".join([line.strip() for line in text.split('\n')])
    return text.strip()


# =====================================================================================
# СТАРТОВАЯ ИНИЦИАЛИЗАЦИЯ
# =====================================================================================

# Загружаем конфигурацию из файла conf.txt
config = load_config('conf.txt')
# Загружаем системный промпт из файла prompt.txt
SYSTEM_PROMPT = load_system_prompt('prompt.txt')

# --- Получение настроек из загруженной конфигурации ---
try:
    TELEGRAM_BOT_TOKEN = config['TELEGRAM_BOT_TOKEN']
    TARGET_GROUP_ID = int(config['TARGET_GROUP_ID'])
    AITUNNEL_API_KEY = config['AITUNNEL_API_KEY']
    AITUNNEL_MODEL = config.get('AITUNNEL_MODEL', 'gpt-4o-search-preview')
    AITUNNEL_API_URL = config.get('AITUNNEL_API_URL', 'https://api.aitunnel.ru/v1/chat/completions')
    AITUNNEL_TRANSCRIBE = config.get('AITUNNEL_TRANSCRIBE', 'whisper-1')
    HISTORY_LIMIT = int(config.get('HISTORY_LIMIT', 20))
    STATE_FILE = config.get('STATE_FILE', 'temp.txt')
except KeyError as e:
    print(f"ERROR: Отсутствует обязательный параметр в файле конфигурации (conf.txt): {e}")
    sys.exit()
except ValueError as e:
    print(f"ERROR: Некорректное значение параметра в файле конфигурации (conf.txt): {e}")
    sys.exit()

# Проверяем, что промпт не пустой после загрузки
if not SYSTEM_PROMPT:
     print("ERROR: Файл системного промпта пустой. Пожалуйста, заполните его.")
     sys.exit()

# --- Загрузка состояния бота из файла при старте ---
load_state(STATE_FILE)


# =====================================================================================
# ОБРАБОТЧИКИ КОМАНД И СООБЩЕНИЙ TELEGRAM
# =====================================================================================

async def role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатывает команду /role.
    Устанавливает постоянную роль (имя) для пользователя и сохраняет ее в файле состояния.
    """
    chat_id = update.effective_chat.id

    # Игнорируем команду, если она вызвана не в целевой группе
    if chat_id != TARGET_GROUP_ID:
        return

    # context.args - это список слов, идущих после команды
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /role <Имя>")
        return

    role_name = " ".join(args).strip()
    if not role_name:
        await update.message.reply_text("Использование: /role <Имя>")
        return

    user_id = str(update.effective_user.id) # user_id пользователя, отправившего команду

    # Убедимся, что структура для этого чата существует в нашем состоянии
    ensure_chat_state_exists(chat_id)

    # Обновляем роль в глобальном словаре состояния
    bot_state['chats'][str(chat_id)]['roles'][user_id] = role_name
    logger.info(f"Роль '{role_name}' установлена для пользователя {user_id} в чате {chat_id}")

    # Сохраняем все состояние в файл
    save_state(STATE_FILE)

    # Отправляем подтверждение пользователю
    await update.message.reply_text(f"Ваша роль установлена как '{role_name}'.")

async def request_clear_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатывает команду /clear.
    Не очищает историю сразу, а отправляет сообщение с кнопками для подтверждения.
    """
    chat_id = update.effective_chat.id

    if chat_id != TARGET_GROUP_ID:
        return

    # Создаем инлайн-кнопки
    keyboard = [
        [
            # "Да": callback_data - это уникальная строка, которую бот получит при нажатии.
            InlineKeyboardButton("Да, очистить", callback_data='clear_confirm_yes'),
            # "Нет":
            InlineKeyboardButton("Нет, отмена", callback_data='clear_confirm_no')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Отправляем сообщение с запросом и прикрепляем кнопки
    message = await update.message.reply_text("Вы уверены, что хотите очистить историю диалога?", reply_markup=reply_markup)

    # Сохраняем ID этого сообщения во временном хранилище контекста чата.
    # Это нужно, чтобы в дальнейшем обработать нажатие кнопки именно под ЭТИМ сообщением
    # и игнорировать нажатия на кнопки под старыми сообщениями.
    context.chat_data['clear_confirmation_message_id'] = message.message_id
    logger.info(f"Отправлен запрос на подтверждение очистки (ID сообщения: {message.message_id}) для чата {chat_id}")

async def handle_clear_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатывает нажатия на инлайн-кнопки подтверждения очистки истории.
    Вызывается, когда пользователь нажимает "Да" или "Нет".
    """
    query = update.callback_query # Объект, содержащий информацию о нажатии
    chat_id = query.message.chat.id
    message_id = query.message.message_id
    callback_data = query.data # 'clear_confirm_yes' или 'clear_confirm_no'

    # Обязательно "отвечаем" на запрос, чтобы убрать "часики" на кнопке в интерфейсе Telegram
    await query.answer()

    # Проверяем, что нажатие произошло в целевой группе
    if chat_id != TARGET_GROUP_ID:
        logger.warning(f"Получено нажатие подтверждения из другого чата {chat_id}. Игнорируется.")
        return

    # Проверяем, что нажата кнопка под актуальным сообщением подтверждения
    expected_message_id = context.chat_data.get('clear_confirmation_message_id')
    if expected_message_id is None or message_id != expected_message_id:
        logger.warning(f"Получено нажатие для устаревшего сообщения (ID: {message_id}, ожидалось: {expected_message_id}).")
        # Редактируем старое сообщение, чтобы пользователь знал, что оно больше неактивно
        try:
             await query.edit_message_text("Это подтверждение устарело. Вызовите /clear снова.")
        except Exception as e:
             logger.warning(f"Не удалось отредактировать устаревшее сообщение {message_id}: {e}")
        return

    # Удаляем ID из временного хранилища, так как мы обработали этот запрос
    del context.chat_data['clear_confirmation_message_id']

    # Обрабатываем выбор пользователя
    if callback_data == 'clear_confirm_yes':
        # Пользователь нажал "Да"
        ensure_chat_state_exists(chat_id)
        bot_state['chats'][str(chat_id)]['history'] = [] # Очищаем историю в постоянном состоянии

        # Очищаем также временный буфер сообщений
        if 'current_batch_messages' in context.chat_data:
            context.chat_data['current_batch_messages'] = []

        save_state(STATE_FILE) # Сохраняем изменения в файл

        logger.info(f"История для чата {chat_id} очищена по подтверждению пользователя.")
        # Редактируем исходное сообщение, чтобы убрать кнопки и показать результат
        await query.edit_message_text("История диалога очищена.")

    elif callback_data == 'clear_confirm_no':
        # Пользователь нажал "Нет"
        logger.info(f"Очистка истории для чата {chat_id} отменена пользователем.")
        await query.edit_message_text("Очистка истории отменена.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатывает обычные текстовые сообщения (не команды).
    Форматирует текст, добавляя роль, и сохраняет его во временный буфер для последующей отправки.
    """
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id)
    text = update.message.text

    if chat_id != TARGET_GROUP_ID or not text:
        return

    # Убедимся, что структура для этого чата существует в нашем состоянии
    ensure_chat_state_exists(chat_id)

    # Получаем роль пользователя из постоянного состояния бота
    role_name = bot_state['chats'][str(chat_id)]['roles'].get(user_id, None)

    if role_name:
        # Если роль задана, используем ее
        formatted_text = f"{role_name}: {text}"
    else:
        # Если роль не задана, используем имя пользователя из Telegram
        user_name = update.effective_user.first_name
        if update.effective_user.last_name:
             user_name += f" {update.effective_user.last_name}"
        formatted_text = f"{user_name}: {text}"

    # Сохраняем отформатированное сообщение во временный буфер `context.chat_data`.
    # Этот буфер сбрасывается после каждого вызова /send и не сохраняется в файле состояния.
    # Он нужен для накопления сообщений между вызовами /send.
    if 'current_batch_messages' not in context.chat_data:
        context.chat_data['current_batch_messages'] = []

    # Сохраняем сообщение в формате, удобном для нашей внутренней логики
    context.chat_data['current_batch_messages'].append({'role': 'user', 'content': formatted_text})
    logger.info(f"Сообщение добавлено в буфер от пользователя {user_id}: {formatted_text}")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатывает голосовые сообщения (voice) Telegram:
    - Скачивает файл
    - Отправляет на AITUNNEL (whisper-1)
    - Добавляет расшифровку в буфер сообщений как обычный текст
    """
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id)
    voice = update.message.voice

    if chat_id != TARGET_GROUP_ID or not voice:
        return

    ensure_chat_state_exists(chat_id)

    # Получаем роль пользователя
    role_name = bot_state['chats'][str(chat_id)]['roles'].get(user_id, None)
    if role_name:
        prefix = f"{role_name}: "
    else:
        user_name = update.effective_user.first_name
        if update.effective_user.last_name:
            user_name += f" {update.effective_user.last_name}"
        prefix = f"{user_name}: "

    # Скачиваем файл
    file = await context.bot.get_file(voice.file_id)
    file_path = f"voice_{voice.file_id}.ogg"
    await file.download_to_drive(file_path)

    # Готовим запрос к AITUNNEL whisper-1
    aitunnel_url = 'https://api.aitunnel.ru/v1/audio/transcriptions'  # URL для транскрипции фиксирован
    aitunnel_key = config.get('AITUNNEL_API_KEY')
    aitunnel_model = config.get('AITUNNEL_TRANSCRIBE', 'whisper-1')
    headers = {"Authorization": f"Bearer {aitunnel_key}"}
    files = {
        'file': (file_path, open(file_path, 'rb'), 'audio/ogg'),
        'model': (None, aitunnel_model)
    }
    try:
        response = requests.post(aitunnel_url, headers=headers, files=files)
        response.raise_for_status()
        data = response.json()
        text = data.get('text', '').strip()
        if not text:
            await update.message.reply_text("Не удалось расшифровать голосовое сообщение.")
            return
        formatted_text = prefix + text
        if 'current_batch_messages' not in context.chat_data:
            context.chat_data['current_batch_messages'] = []
        context.chat_data['current_batch_messages'].append({'role': 'user', 'content': formatted_text})
        logger.info(f"Голосовое сообщение расшифровано и добавлено в буфер: {formatted_text}")
        await update.message.reply_text(f"Распознано: {text}")
    except Exception as e:
        logger.error(f"Ошибка при расшифровке голосового сообщения: {e}", exc_info=True)
        await update.message.reply_text(f"Ошибка при расшифровке голосового сообщения: {e}")
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

async def send_to_aitunnel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатывает команду /send.
    Собирает историю и новые сообщения, отправляет их в AITUNNEL, получает ответ и сохраняет историю.
    """
    if OpenAI is None:
        await update.message.reply_text("Модуль openai не установлен. Установите его: pip install openai")
        return
    chat_id = update.effective_chat.id
    if chat_id != TARGET_GROUP_ID:
        return
    ensure_chat_state_exists(chat_id)
    chat_state = bot_state['chats'][str(chat_id)]
    history = chat_state.get('history', [])
    current_batch_messages = context.chat_data.get('current_batch_messages', [])
    if not current_batch_messages:
        await update.message.reply_text("Нет новых сообщений для отправки в нейросеть.")
        return
    # Формируем список сообщений для AITUNNEL
    messages_for_api = []
    for msg in history:
        if not msg.get('content'): continue
        messages_for_api.append({"role": msg['role'], "content": msg['content']})
    for msg in current_batch_messages:
        if not msg.get('content'): continue
        messages_for_api.append({"role": msg['role'], "content": msg['content']})
    # Ограничиваем историю
    if len(messages_for_api) > HISTORY_LIMIT:
        messages_for_api = messages_for_api[-HISTORY_LIMIT:]
    try:
        client = OpenAI(api_key=AITUNNEL_API_KEY, base_url="https://api.aitunnel.ru/v1/")
        loop = asyncio.get_event_loop()
        # OpenAI SDK синхронный, поэтому используем run_in_executor
        def do_request():
            return client.chat.completions.create(
                messages=messages_for_api,
                model=AITUNNEL_MODEL,
                max_tokens=2048
            )
        chat_result = await loop.run_in_executor(None, do_request)
        ai_text = chat_result.choices[0].message.content.strip()
        await update.message.reply_text(ai_text)
        # Сохраняем в историю
        chat_state['history'].extend(current_batch_messages)
        chat_state['history'].append({'role': 'assistant', 'content': ai_text})
        if len(chat_state['history']) > HISTORY_LIMIT:
            chat_state['history'] = chat_state['history'][-HISTORY_LIMIT:]
        context.chat_data['current_batch_messages'] = []
        save_state(STATE_FILE)
        logger.info(f"Ответ AITUNNEL отправлен и сохранён в истории.")
    except Exception as e:
        logger.error(f"Ошибка при обращении к AITUNNEL: {e}", exc_info=True)
        await update.message.reply_text(f"Ошибка при обращении к AITUNNEL: {e}")


# =====================================================================================
# ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА БОТА
# =====================================================================================

def main() -> None:
    """
    Основная функция, которая создает, настраивает и запускает Telegram бота.
    """
    # Создаем объект Application, который является "сердцем" бота
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Регистрируем обработчики ---
    # Каждый обработчик связывает определенное событие (команду, сообщение, нажатие кнопки)
    # с функцией, которая должна его обработать.

    # Команда /role вызывает функцию role()
    application.add_handler(CommandHandler("role", role))
    # Команда /clear вызывает функцию request_clear_confirmation()
    application.add_handler(CommandHandler("clear", request_clear_confirmation))

    # Нажатие на инлайн-кнопку, у которой callback_data начинается с 'clear_confirm_',
    # вызывает функцию handle_clear_confirmation().
    application.add_handler(CallbackQueryHandler(handle_clear_confirmation, pattern='^clear_confirm_'))

    # Обработчик для текстовых сообщений, которые не являются командами, и только в целевой группе.
    # Вызывает функцию handle_message().
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Chat(chat_id=TARGET_GROUP_ID), handle_message))

    # Обработчик для голосовых сообщений, которые только в целевой группе.
    # Вызывает функцию handle_voice_message().
    application.add_handler(MessageHandler(filters.VOICE & filters.Chat(chat_id=TARGET_GROUP_ID), handle_voice_message))

    # Команда /send вызывает функцию send_to_aitunnel()
    application.add_handler(CommandHandler("send", send_to_aitunnel))

    # --- Запуск бота ---
    logger.info("Бот запущен и готов к работе...")
    # Запускаем бота в режиме "polling" - он сам будет периодически опрашивать Telegram о новых сообщениях.
    # allowed_updates - это оптимизация: мы говорим Telegram присылать нам только обновления
    # о новых сообщениях и нажатиях на инлайн-кнопки, игнорируя все остальное.
    application.run_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY])

if __name__ == "__main__":
    # Этот стандартный блок Python гарантирует, что функция main() будет вызвана
    # только тогда, когда скрипт запускается напрямую, а не когда он импортируется как модуль.
    main()
