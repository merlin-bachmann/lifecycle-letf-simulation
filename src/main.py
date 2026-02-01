import asyncio
import logging

import polars as pl

pl.Config.set_tbl_formatting("UTF8_FULL", True)
pl.Config.set_tbl_cols(150)
pl.Config.set_tbl_width_chars(4000)
pl.Config.set_tbl_rows(20)
pl.Config.set_fmt_str_lengths(500)

logging.basicConfig(level=logging.INFO, format="%(asctime)s   %(levelname)s   %(message)s")
logging.getLogger("httpx").setLevel(logging.ERROR)


async def main(): ...


if __name__ == "__main__":
    asyncio.run(main())
