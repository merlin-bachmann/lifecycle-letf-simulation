# pyright: reportUnusedFunction=false
import logging

import polars as pl
from fastapi import FastAPI, Response

pl.Config.set_tbl_formatting("UTF8_FULL", True)
pl.Config.set_tbl_cols(150)
pl.Config.set_tbl_width_chars(4000)
pl.Config.set_tbl_rows(20)
pl.Config.set_fmt_str_lengths(500)

logging.basicConfig(level=logging.INFO, format="%(asctime)s   %(levelname)s   %(message)s")
logging.getLogger("httpx").setLevel(logging.ERROR)

app = FastAPI()
# app.mount("/css", StaticFiles(directory="static/css"), name="css")
# app.mount("/img", StaticFiles(directory="static/img"), name="img")


@app.get("/status")
async def _run_all_transfers() -> Response:
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
