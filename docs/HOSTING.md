# Host a DerivPort demo

Streamlit Community Cloud is a free hosting option that runs this Python app directly from GitHub. See the [official service overview](https://docs.streamlit.io/deploy/streamlit-community-cloud).

## Deploy

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that owns or administers `bvnnydevv/DerivPort`.
2. Choose the **bvnnydevv** workspace, then **Create app** and **Yup, I have an app**.
3. Enter these values:

   | Setting | Value |
   | --- | --- |
   | Repository | `bvnnydevv/DerivPort` |
   | Branch | `main` |
   | Main file path | `app.py` |

   Alternatively, paste [the app's GitHub URL](https://github.com/bvnnydevv/DerivPort/blob/main/app.py) into **Paste GitHub URL**.

4. Open **Advanced settings**, select Python **3.13**, and save. This matches the Python version used to validate the app. Leave secrets empty; the bundled demo does not require any.
5. Optionally choose an available subdomain, then click **Deploy**. Share the resulting `streamlit.app` URL after the app finishes launching.

The repository already includes `requirements.txt`, the Streamlit theme, local Python modules, company notes, and bundled price data. Community Cloud installs the declared dependencies. Follow the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) if the interface changes.

## Check the live demo

- Open **Research** with AAPL and **Full cached history**. The supplied cache ends on **2026-11-13**; the latest cutoff correctly shows a pending 21-session outcome.
- Verify real candlesticks, weekly/monthly intervals, Customize and Fullscreen. The full 6,094-listing directory and 15 bundled candle histories work offline; new stocks load their histories on selection.
- Apply a guided recipe (or choose Advanced blocks to drag factors), and ask the stock-panel coach to mark a chart guide. The offline guide needs no key. For AI-generated replies, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in app secrets; never commit those credentials.
- Select an earlier cutoff to demonstrate historical outcome reveal.
- In **Algorithm builder**, click **Use my balanced algorithm**, open the calculus lab, and run a historical backtest.
- In **Research & Benchmarking**, select development and final-test dates, then run the experiment. This heavier model comparison runs only when requested; prepare it before presenting.
- In **Future scenarios**, compare Low/Medium/High assumptions and the purple scenario fan with the amber terminal-return distribution. Dates through 2037 are hypothetical simulations.

The bundled stocks work without a market-data request. Listing searches, uncached stocks, and refreshes use the external provider. New downloads belong to the running host's local cache; retain important demonstration data in the repository rather than relying on that runtime cache surviving a restart.

## Keep it ready

Changes pushed to the deployed branch trigger app updates. Community Cloud apps can sleep after inactivity, so open the URL and allow startup to complete before presenting. Use **Manage app** to view deployment logs or reboot the app if needed. See [Streamlit's app management guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

The local `127.0.0.1:8501` preview is available only on the computer running it. Use the deployed `streamlit.app` link for judges and other viewers.
