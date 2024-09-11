import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint
from fastapi import FastAPI
from Server.config.basic_config import LOG_PATH, log_verbose

try:
    import numexpr # 加速库

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def create_model_worker_app(log_level: str="INFO", **kwargs) -> FastAPI:
    pass

def run_webui(started_event: mp.Event = None, run_mode: str = None): 
    from Server.utils import set_httpx_config
    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    cmd = ["streamlit", "run", "webui.py",
           "--server.address", host,
           "--server.port", str(port),
           "--theme.base", "light",
           "--theme.primaryColor", "#165dff",
           "--theme.secondaryBackgroundColor", "#f5f5f5",
           "--theme.textColor", "#000000",
           ]
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]
    p = subprocess.Popen(cmd)
    started_event.set()
    p.wait()


async def start_main_server():
    import time
    import signal

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    processes = {}
    webui_started = manager.Event()
    process = Process(
        target=run_webui,
        name=f"WEBUI Server",
        kwargs=dict(started_event=webui_started, run_mode=run_mode),
        daemon=True,
    )
    processes["webui"] = process

