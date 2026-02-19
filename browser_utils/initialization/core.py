# --- browser_utils/initialization/core.py ---
import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from playwright.async_api import (
    Browser as AsyncBrowser,
)
from playwright.async_api import (
    BrowserContext as AsyncBrowserContext,
)
from playwright.async_api import (
    Error as PlaywrightAsyncError,
)
from playwright.async_api import (
    Page as AsyncPage,
)
from playwright.async_api import (
    expect as expect_async,
)

from config import (
    AI_STUDIO_URL_PATTERN,
    INPUT_SELECTOR,
    MODEL_NAME_SELECTOR,
    USER_INPUT_END_MARKER_SERVER,
    USER_INPUT_START_MARKER_SERVER,
)
from config.selector_utils import (
    INPUT_WRAPPER_SELECTORS,
)

from .auth import wait_for_model_list_and_handle_auth_save
from .debug import setup_debug_listeners
from .network import setup_network_interception_and_scripts

logger = logging.getLogger("AIStudioProxyServer")


async def initialize_page_logic(  # pragma: no cover
    browser: AsyncBrowser, storage_state_path: Optional[str] = None
) -> Tuple[AsyncPage, bool]:
    """
    初始化页面逻辑，连接到现有浏览器

    Args:
        browser: Playwright 浏览器实例
        storage_state_path: 可选的认证文件路径。如果提供，将优先使用此路径。
    """
    logger.debug("[Init] Initializing page logic")
    temp_context: Optional[AsyncBrowserContext] = None
    storage_state_path_to_use: Optional[str] = None
    launch_mode = os.environ.get("LAUNCH_MODE", "debug")
    loop = asyncio.get_running_loop()

    # 优先使用传入的 storage_state_path
    if storage_state_path:
        if os.path.exists(storage_state_path):
            storage_state_path_to_use = storage_state_path
            logger.debug(f"Using specified auth file: {storage_state_path_to_use}")
        else:
            logger.error(f"Specified auth file does not exist: {storage_state_path}")
            # 如果是明确指定的路径但不存在，这应该是一个错误
            raise RuntimeError(f"Specified auth file does not exist: {storage_state_path}")
    else:
        # 回退到原有的环境变量逻辑
        if launch_mode == "headless" or launch_mode == "virtual_headless":
            auth_filename = os.environ.get("ACTIVE_AUTH_JSON_PATH")
            if auth_filename:
                constructed_path = auth_filename
                if os.path.exists(constructed_path):
                    storage_state_path_to_use = constructed_path
                else:
                    logger.error(
                        f"{launch_mode} mode auth file is invalid or missing: '{constructed_path}'"
                    )
                    raise RuntimeError(
                        f"{launch_mode} mode auth file is invalid: '{constructed_path}'"
                    )
            else:
                logger.error(
                    f"{launch_mode} mode requires ACTIVE_AUTH_JSON_PATH, but it is missing or empty."
                )
                raise RuntimeError(f"{launch_mode} mode requires ACTIVE_AUTH_JSON_PATH.")
        elif launch_mode == "debug":
            logger.info(
                "Debug mode: trying to load auth file from ACTIVE_AUTH_JSON_PATH..."
            )
            auth_filepath_from_env = os.environ.get("ACTIVE_AUTH_JSON_PATH")
            if auth_filepath_from_env and os.path.exists(auth_filepath_from_env):
                storage_state_path_to_use = auth_filepath_from_env
                logger.debug(
                    f"Debug mode auth file from environment: {storage_state_path_to_use}"
                )
            elif auth_filepath_from_env:
                logger.warning(
                    f"In debug mode, ACTIVE_AUTH_JSON_PATH points to a missing file: '{auth_filepath_from_env}'. Skipping auth file load."
                )
            else:
                logger.info(
                    "No auth file provided via environment in debug mode. Using current browser state."
                )
        elif launch_mode == "direct_debug_no_browser":
            logger.info(
                "direct_debug_no_browser mode: no storage_state loaded, no browser operations performed."
            )
        else:
            logger.warning(
                f"Warning: unknown launch mode '{launch_mode}'. Skipping storage_state load."
            )

    try:
        # Consolidate into one log message
        auth_file = (
            os.path.basename(storage_state_path_to_use)
            if storage_state_path_to_use
            else None
        )
        context_options: Dict[str, Any] = {"viewport": {"width": 460, "height": 800}}
        if storage_state_path_to_use:
            context_options["storage_state"] = storage_state_path_to_use

        # 代理设置需要从 server 模块中获取
        import server

        if server.PLAYWRIGHT_PROXY_SETTINGS:
            context_options["proxy"] = server.PLAYWRIGHT_PROXY_SETTINGS
            logger.debug(
                f"[Browser] Context configured with proxy: {server.PLAYWRIGHT_PROXY_SETTINGS['server']}"
            )

        context_options["ignore_https_errors"] = True

        # Single consolidated log
        if auth_file:
            logger.info(f"[Browser] Context created (Auth: {auth_file})")
        else:
            logger.debug("[Browser] Context created (no auth)")

        temp_context = await browser.new_context(**context_options)

        # 设置网络拦截和脚本注入
        await setup_network_interception_and_scripts(temp_context)

        found_page: Optional[AsyncPage] = None
        pages = temp_context.pages
        target_url_base = f"https://{AI_STUDIO_URL_PATTERN}"
        target_full_url = f"{target_url_base}prompts/new_chat"
        login_url_pattern = "accounts.google.com"
        current_url = ""

        # 导入_handle_model_list_response - 需要延迟导入避免循环引用
        from browser_utils.operations import _handle_model_list_response

        for p_iter in pages:
            try:
                page_url_to_check = p_iter.url
                if (
                    not p_iter.is_closed()
                    and target_url_base in page_url_to_check
                    and "/prompts/" in page_url_to_check
                ):
                    found_page = p_iter
                    current_url = page_url_to_check
                    logger.debug(f"Found existing AI Studio page: {current_url}")
                    if found_page:
                        logger.debug(
                            f"Added model list response listener for existing page: {found_page.url}."
                        )
                        found_page.on("response", _handle_model_list_response)
                        # Setup debug listeners for error snapshots
                        setup_debug_listeners(found_page)
                    break
            except PlaywrightAsyncError as pw_err_url:
                logger.warning(f"Playwright error while checking page URL: {pw_err_url}")
            except AttributeError as attr_err_url:
                logger.warning(f"Attribute error while checking page URL: {attr_err_url}")
            except asyncio.CancelledError:
                raise
            except Exception as e_url_check:
                logger.warning(
                    f"Unexpected error while checking page URL: {e_url_check} (type: {type(e_url_check).__name__})"
                )

        if not found_page:
            logger.info(f"[Navigation] Opening new page: {target_full_url}")
            found_page = await temp_context.new_page()
            if found_page:
                logger.debug("Added model list response listener for newly created page (before navigation).")
                found_page.on("response", _handle_model_list_response)
                # Setup debug listeners for error snapshots
                setup_debug_listeners(found_page)
            try:
                await found_page.goto(
                    target_full_url, wait_until="domcontentloaded", timeout=90000
                )
                current_url = found_page.url
                logger.debug(f"New page navigation attempt completed. Current URL: {current_url}")
            except asyncio.CancelledError:
                raise
            except Exception as new_page_nav_err:
                # 导入save_error_snapshot函数
                from browser_utils.operations import save_error_snapshot

                await save_error_snapshot("init_new_page_nav_fail")
                error_str = str(new_page_nav_err)
                if "NS_ERROR_NET_INTERRUPT" in error_str:
                    logger.error("\n" + "=" * 30 + " Network Navigation Error " + "=" * 30)
                    logger.error(
                        f"Navigation to '{target_full_url}' failed with network interruption error (NS_ERROR_NET_INTERRUPT)."
                    )
                    logger.error("This usually means the browser connection was unexpectedly interrupted while loading the page.")
                    logger.error("Possible causes and troubleshooting steps:")
                    logger.error(
                        "     1. Network: verify local network stability and try opening the target URL in a regular browser."
                    )
                    logger.error(
                        "     2. AI Studio service: confirm aistudio.google.com is available."
                    )
                    logger.error(
                        "     3. Firewall/proxy/VPN: check local firewall, antivirus, proxy, or VPN settings."
                    )
                    logger.error(
                        "     4. Camoufox service: ensure launch_camoufox.py is running correctly."
                    )
                    logger.error(
                        "     5. System resources: ensure sufficient memory and CPU are available."
                    )
                    logger.error("=" * 74 + "\n")
                raise RuntimeError(
                    f"Failed to navigate new page: {new_page_nav_err}"
                ) from new_page_nav_err

        if login_url_pattern in current_url:
            if launch_mode == "headless":
                logger.error(
                    "Headless mode was redirected to login page. Auth may be expired; please update auth file."
                )
                raise RuntimeError("Headless authentication failed; auth file update required.")
            else:
                print(f"\n{'=' * 20} Action Required {'=' * 20}", flush=True)
                login_prompt = "   Login may be required. If the browser shows a login page, complete Google login in the browser window, then press Enter here to continue..."
                # NEW: If SUPPRESS_LOGIN_WAIT is set, skip waiting for user input.
                if os.environ.get("SUPPRESS_LOGIN_WAIT", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    logger.info("SUPPRESS_LOGIN_WAIT detected, skipping user input wait.")
                else:
                    print(USER_INPUT_START_MARKER_SERVER, flush=True)
                    await loop.run_in_executor(None, input, login_prompt)
                    print(USER_INPUT_END_MARKER_SERVER, flush=True)
                logger.info("Checking login status...")
                try:
                    await found_page.wait_for_url(
                        f"**/{AI_STUDIO_URL_PATTERN}**", timeout=180000
                    )
                    current_url = found_page.url
                    if login_url_pattern in current_url:
                        logger.error("After manual login attempt, page still appears to be on login screen.")
                        raise RuntimeError("Still on login page after manual login attempt.")
                    logger.info("Login successful. Please do not interact with the browser window and wait for next steps.")

                    # 登录成功后，调用认证保存逻辑
                    if os.environ.get("AUTO_SAVE_AUTH", "false").lower() == "true":
                        await wait_for_model_list_and_handle_auth_save(
                            temp_context, launch_mode, loop
                        )

                except asyncio.CancelledError:
                    raise
                except Exception as wait_login_err:
                    from browser_utils.operations import save_error_snapshot

                    await save_error_snapshot("init_login_wait_fail")
                    logger.error(
                        f"Failed to detect AI Studio URL after login prompt, or error occurred during state save: {wait_login_err}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Failed to detect AI Studio URL after login prompt: {wait_login_err}"
                    ) from wait_login_err

        elif target_url_base not in current_url or "/prompts/" not in current_url:
            from browser_utils.operations import save_error_snapshot

            await save_error_snapshot("init_unexpected_page")
            logger.error(
                f"Unexpected page URL after initial navigation: {current_url}. Expected URL to include '{target_url_base}' and '/prompts/'."
            )
            raise RuntimeError(f"Unexpected page after initial navigation: {current_url}.")

        await found_page.bring_to_front()

        try:
            # 使用集中管理的选择器回退逻辑查找输入容器
            # 支持当前和旧 UI 结构 (ms-prompt-input-wrapper / ms-chunk-editor / ms-prompt-box)
            # 使用 find_first_visible_locator 等待元素可见，解决无头模式下的时序问题
            from config.selector_utils import find_first_visible_locator

            (
                input_wrapper_locator,
                matched_selector,
            ) = await find_first_visible_locator(
                found_page,
                INPUT_WRAPPER_SELECTORS,
                description="input container",
                timeout_per_selector=30000,  # 每个选择器等待30秒
            )
            if not input_wrapper_locator:
                raise RuntimeError(
                    "Unable to locate input container element. Selectors tried: "
                    + ", ".join(INPUT_WRAPPER_SELECTORS)
                )
            # 容器已通过 find_first_visible_locator 确认可见，直接检查输入框
            await expect_async(found_page.locator(INPUT_SELECTOR)).to_be_visible(
                timeout=10000
            )
            logger.debug(f"[Selector] Input area located and visible ({matched_selector})")

            model_name_locator = found_page.locator(MODEL_NAME_SELECTOR)
            try:
                model_name_on_page = await model_name_locator.first.inner_text(
                    timeout=5000
                )
            except PlaywrightAsyncError as e:
                logger.error(f"Error while getting model name (model_name_locator): {e}")
                raise

            result_page_instance = found_page
            result_page_ready = True

            logger.info(f"[Page] Initialization successful | Current model: {model_name_on_page}")
            return result_page_instance, result_page_ready
        except asyncio.CancelledError:
            raise
        except Exception as input_visible_err:
            from browser_utils.operations import save_error_snapshot

            await save_error_snapshot("init_fail_input_timeout")
            logger.error(
                f"Page initialization failed: core input area did not become visible in time. Last URL: {found_page.url}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Page initialization failed: core input area did not become visible in time. Last URL: {found_page.url}"
            ) from input_visible_err
    except asyncio.CancelledError:
        logger.warning("Page initialization cancelled.")
        if temp_context:
            try:
                await temp_context.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        raise
    except Exception as e_init_page:
        logger.critical(
            f"Critical unexpected error during page logic initialization: {e_init_page}", exc_info=True
        )
        if temp_context:
            try:
                logger.info(
                    "Trying to close temporary browser context due to initialization error."
                )
                await temp_context.close()
                logger.info("Temporary browser context closed.")
            except asyncio.CancelledError:
                raise
            except Exception as close_err:
                logger.warning(f"Error while closing temporary browser context: {close_err}")
        from browser_utils.operations import save_error_snapshot

        await save_error_snapshot("init_unexpected_error")
        raise RuntimeError(f"Unexpected page initialization error: {e_init_page}") from e_init_page


async def close_page_logic() -> Tuple[None, bool]:  # pragma: no cover
    """Close page logic."""
    # 需要访问全局变量
    from api_utils.server_state import state

    logger.info("--- Running page close logic --- ")
    if state.page_instance and not state.page_instance.is_closed():
        try:
            await state.page_instance.close()
            logger.info("Page closed")
        except PlaywrightAsyncError as pw_err:
            logger.warning(f"Playwright error while closing page: {pw_err}")
        except asyncio.TimeoutError as timeout_err:
            logger.warning(f"Timeout while closing page: {timeout_err}")
        except asyncio.CancelledError:
            raise
        except Exception as other_err:
            logger.error(
                f"Unexpected error while closing page: {other_err} (type: {type(other_err).__name__})",
                exc_info=True,
            )
    state.page_instance = None
    state.is_page_ready = False
    logger.info("Page logic state reset.")
    return None, False


async def signal_camoufox_shutdown() -> None:  # pragma: no cover
    """Send shutdown signal to Camoufox server."""
    logger.info("Attempting to send shutdown signal to Camoufox server (may already be handled by parent process)...")
    ws_endpoint = os.environ.get("CAMOUFOX_WS_ENDPOINT")
    if not ws_endpoint:
        logger.warning("Cannot send shutdown signal: CAMOUFOX_WS_ENDPOINT environment variable not found.")
        return

    # 需要访问全局浏览器实例
    import server

    if not server.browser_instance or not server.browser_instance.is_connected():
        logger.warning("Browser instance is disconnected or uninitialized; skipping shutdown signal.")
        return
    try:
        await asyncio.sleep(0.2)
        logger.info("(Simulated) Shutdown signal handled.")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Exception while sending shutdown signal: {e}", exc_info=True)


async def enable_temporary_chat_mode(page: AsyncPage) -> None:  # pragma: no cover
    """
    Check and enable "Temporary chat" mode in AI Studio UI.
    This is an independent UI action and should be called after page stabilization.
    """
    try:
        incognito_button_locator = page.locator(
            'button[aria-label="Temporary chat toggle"]'
        )

        await incognito_button_locator.wait_for(state="visible", timeout=10000)

        button_classes = await incognito_button_locator.get_attribute("class")

        if button_classes and "ms-button-active" in button_classes:
            logger.debug("[UI] Temporary chat mode already active")
        else:
            await incognito_button_locator.click(timeout=5000, force=True)
            await asyncio.sleep(1)

            updated_classes = await incognito_button_locator.get_attribute("class")
            if updated_classes and "ms-button-active" in updated_classes:
                logger.debug("[UI] Temporary chat mode enabled")
            else:
                logger.warning("[UI] Failed to enable temporary chat mode")

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"[UI] Temporary chat mode error: {e}")
