import os
import time
from openai import OpenAI  # 替换 genai 导入
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# 设置日志记录
logger = setup_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")


# 验证环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 DEEPSEEK_API_KEY 环境变量")
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
if not model:
    model = "deepseek-chat"  # 修改默认模型
    logger.info(f"{WAIT_ICON} 使用默认模型: {model}")

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    # base_url="https://api.siliconflow.cn/v1"
    base_url="https://api.deepseek.com"  # DeepSeek API 端点
)
logger.info(f"{SUCCESS_ICON} DeepSeek 客户端初始化成功")



@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 DeepSeek API...")
        logger.debug(f"请求内容: {contents}")
        logger.debug(f"请求配置: {config}")

        messages = [{"role": "user", "content": contents}]
        if config and 'system_instruction' in config:
            messages.insert(0, {"role": "system", "content": config['system_instruction']})

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        response_text = response.choices[0].message.content

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        logger.info(f"响应内容: {response_text[:500]}..." if len(
            str(response_text)) > 500 else f"响应内容: {response_text}")
        
        # 创建一个类似于 genai 响应的对象
        class Response:
            def __init__(self, text):
                self.text = text
        
        return Response(response_text)

    except Exception as e:
        if "AFC is enabled" in str(e):
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {str(e)}")
            time.sleep(5)
            raise e
        logger.error(f"{ERROR_ICON} API 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        if model is None:
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        logger.info(f"{WAIT_ICON} 使用模型: {model}")
        logger.debug(f"消息内容: {messages}")

        for attempt in range(max_retries):
            try:
                # 转换消息格式
                prompt = ""
                system_instruction = None

                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        system_instruction = content
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"

                # 准备配置
                config = {}
                if system_instruction:
                    config['system_instruction'] = system_instruction

                # 调用 API
                response = generate_content_with_retry(
                    model=model,
                    contents=prompt.strip(),
                    config=config
                )

                if response is None:
                    logger.warning(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return None

                # 转换响应格式
                chat_message = ChatMessage(content=response.text)
                chat_choice = ChatChoice(message=chat_message)
                completion = ChatCompletion(choices=[chat_choice])

                logger.debug(f"API 原始响应: {response.text}")
                logger.info(f"{SUCCESS_ICON} 成功获取响应")
                return completion.choices[0].message.content

            except Exception as e:
                logger.error(
                    f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                    return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None
