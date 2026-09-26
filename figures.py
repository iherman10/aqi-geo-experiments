"""Portfolio-style versions of the key charts in analysis.ipynb.

Rebuilds only the data prep those charts need from the local files in data/
(no API calls), then writes PNGs to figures/. Run with: uv run python figures.py
"""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import portfolio_charts as pc

pc.apply()

AQS_NOTE = "Source: EPA Air Quality System, daily PM2.5 AQI"
SSD_NOTE = "Source: NYC DOHMH Syndromic Surveillance, asthma + respiratory chief complaints"

# ---------------------------------------------------------------------------
# Data prep (mirrors analysis.ipynb)
# ---------------------------------------------------------------------------

# CBSA-level AQI
cbsa_raw = pd.read_parquet("data/daily_pm25_top_20_cbsa.parquet", engine="fastparquet")
cbsa_daily = cbsa_raw.groupby(["cbsa", "cbsa_code", "date_local"]).agg(aqi=("aqi", "mean")).reset_index()
cbsa_daily["date_local"] = pd.to_datetime(cbsa_daily["date_local"], format="%Y-%m-%d")
cbsa_daily["cbsa_code"] = cbsa_daily["cbsa_code"].astype(int)

cbsa_region_map = {
    35620: "Northeast", 37980: "Northeast", 14460: "Northeast", 12580: "Northeast", 47900: "Northeast",
    12060: "Southeast", 33100: "Southeast", 45300: "Southeast",
    16980: "Midwest", 19820: "Midwest", 33460: "Midwest",
    19100: "South Central", 26420: "South Central",
    31080: "West", 41860: "West", 41740: "West", 40140: "West", 38060: "West", 19740: "West", 42660: "West",
}
cbsa_daily["region"] = cbsa_daily["cbsa_code"].map(cbsa_region_map)

# Borough-level AQI (Bronx and Queens, gaps interpolated)
site_raw = pd.read_parquet("data/daily_pm25_ny.parquet", engine="fastparquet")
borough_daily = site_raw.groupby(["borough", "date_local"]).agg(aqi=("aqi", "mean")).reset_index()
borough_daily["date_local"] = pd.to_datetime(borough_daily["date_local"], format="%Y-%m-%d")
aqi = borough_daily[borough_daily["borough"].isin(["Bronx", "Queens"])].copy()
aqi = aqi.sort_values(["borough", "date_local"]).rename(columns={"date_local": "date"})
aqi = aqi.set_index("date")
aqi["aqi"] = aqi.groupby("borough")["aqi"].transform(lambda x: x.interpolate(method="time"))
aqi = aqi.reset_index()

# Health outcomes: ED visits by borough
bronx_zips = [10467, 10456, 10458, 10453, 10468, 10457, 10452, 10469, 10462, 10466, 10463, 10472, 10460,
              10473, 10451, 10461, 10459, 10465, 10475, 10455, 10454, 10471, 10474, 10470, 10803, 10464, 10499]
queens_zips = [11368, 11385, 11373, 11377, 11355, 11375, 11691, 11372, 11432, 11434, 11435, 11354, 11420,
               11419, 11374, 11413, 11365, 11357, 11421, 11412, 11367, 11433, 11378, 11379, 11105, 11364,
               11106, 11358, 11418, 11101, 11103, 11422, 11417, 11423, 11414, 11369, 11361, 11001, 11102,
               11429, 11416, 11104, 11356, 11427, 11370, 11428, 11426, 11694, 11436, 11692, 11411, 11360,
               11415, 11362, 11004, 11693, 11366, 11096, 11363, 11109, 11697, 11005, 11424, 11430, 11390,
               11371, 11451, 11690, 11695, 11120, 11351, 11352, 11359, 11380, 11381, 11386, 11405, 11425,
               11431, 11439, 11499, 11437]

health_raw = pd.concat([
    pd.read_csv(f"data/health_outcomes/{c}.csv", sep="\t", encoding="utf-16").assign(complaint=c)
    for c in ["asthma", "respiratory"]
])
health = health_raw[health_raw["Zip"].isin(bronx_zips + queens_zips)].copy()
health = health[health["Dim2Value"] == "All age groups"]
health["Zip"] = health["Zip"].astype(int)
health["Date "] = pd.to_datetime(health["Date "], format="%m/%d/%y")
health = health.rename(columns={"Zip": "zip", "Date ": "date", "Count": "count"})
health["count"] = health["count"].astype(int)
health["borough"] = np.where(health["zip"].isin(bronx_zips), "Bronx", "Queens")

# Final ED series, with two missing dates interpolated
health_outcomes = health.groupby(["date", "borough"])["count"].sum().reset_index(name="ed_visits")
missing = pd.DataFrame({"date": pd.to_datetime(["2023-08-28"] * 2 + ["2023-09-17"] * 2),
                        "borough": ["Bronx", "Queens"] * 2, "ed_visits": np.nan})
health_outcomes = pd.concat([health_outcomes, missing], ignore_index=True).set_index("date")
health_outcomes["ed_visits"] = health_outcomes.groupby("borough")["ed_visits"].transform(
    lambda x: x.interpolate(method="time"))
health_outcomes = health_outcomes.reset_index()

# Time-based regression: Queens (treated) predicted from Bronx (control)
TEST_START, TEST_END = pd.Timestamp("2024-02-10"), pd.Timestamp("2024-02-12")
WINDOW = (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-20"))

aqi_final = aqi[aqi["date"].between("2023-02-09", "2024-02-19")].pivot(index="date", columns="borough", values="aqi")
ed_final = health_outcomes[health_outcomes["date"].between("2023-02-09", "2024-02-19")].pivot(
    index="date", columns="borough", values="ed_visits")
tbr = aqi_final.join(ed_final, lsuffix="_aqi", rsuffix="_ed")
tbr.columns = [c.lower() for c in tbr.columns]

post = tbr.index >= TEST_START
pre = tbr[~post]
n_post = np.where(post, np.cumsum(post), 0)

for m in ["aqi", "ed"]:
    model = sm.OLS(pre[f"queens_{m}"], sm.add_constant(pre[f"bronx_{m}"])).fit()
    tbr[f"cf_{m}"] = model.predict(sm.add_constant(tbr[f"bronx_{m}"]))
    resid_std = (tbr.loc[~post, f"queens_{m}"] - tbr.loc[~post, f"cf_{m}"]).std()
    tbr[f"sd_{m}"] = resid_std
    tbr[f"effect_{m}"] = tbr[f"queens_{m}"] - tbr[f"cf_{m}"]
    tbr[f"cum_{m}"] = np.where(post, tbr[f"effect_{m}"], 0).cumsum()
    tbr[f"cum_half_{m}"] = np.where(post, 1.96 * resid_std * np.sqrt(n_post), np.nan)
    tbr[f"half_{m}"] = np.where(post, 1.96 * resid_std, np.nan)

# Effect ratio: incremental ED visits per incremental AQI point (delta-method band, as in the notebook)
cum_ed, cum_aqi = tbr["cum_ed"], tbr["cum_aqi"]
tbr["ratio"] = np.where(cum_aqi != 0, cum_ed / cum_aqi, np.nan)
var_ed = (1.96 * tbr["sd_ed"] * np.sqrt(n_post)) ** 2
var_aqi = (1.96 * tbr["sd_aqi"] * np.sqrt(n_post)) ** 2
with np.errstate(divide="ignore", invalid="ignore"):
    tbr["ratio_half"] = np.sqrt(var_ed / cum_aqi**2 + cum_ed**2 * var_aqi / cum_aqi**4)

win = tbr.loc[WINDOW[0]:WINDOW[1]]
blue, orange = pc.palette(2)


def mask_pre(s):
    """Intervals exist only from the test start onward."""
    return s.where(win.index >= TEST_START)


# ---------------------------------------------------------------------------
# 1. Regional AQI, 20 CBSAs (notebook cell 40)
# ---------------------------------------------------------------------------

regions = ["Northeast", "Southeast", "Midwest", "South Central", "West"]
fig, axs = pc.figure(len(regions), 1, aspect=1.15, sharex=True, sharey=True)
for ax, region in zip(axs, regions):
    rd = cbsa_daily[cbsa_daily["region"] == region]
    peak = rd.loc[rd["aqi"].idxmax()]
    for name, g in rd.groupby("cbsa"):
        if name != peak["cbsa"]:
            ax.plot(g["date_local"], g["aqi"], color=pc.context(), lw=0.5, zorder=1)
    g = rd[rd["cbsa"] == peak["cbsa"]]
    ax.plot(g["date_local"], g["aqi"], color=blue, lw=0.7, zorder=2)
    city = peak["cbsa"].split("-")[0].split(",")[0]
    ax.annotate(f"{city}: {peak['aqi']:.0f} on {peak['date_local']:%b %-d, %Y}",
                xy=(peak["date_local"], peak["aqi"]), xytext=(6, -2), textcoords="offset points",
                va="top", fontsize=8, color=pc.ink("primary"))
    ax.set_title(f"{region} ({rd['cbsa'].nunique()} metros)")
    ax.set_yticks([0, 100, 200])
axs[len(regions) // 2].set_ylabel("Daily AQI")
pc.title(fig, "The biggest AQI spikes hit the West, Midwest and Northeast",
         "Daily mean PM2.5 AQI, 20 largest US metros, 2020–2024. Blue: the metro with each region's peak day")
pc.note(fig, AQS_NOTE)
pc.save(fig, "figures/regional-aqi")

# ---------------------------------------------------------------------------
# 2. NYC seasonal pattern by year (notebook cell 49)
# ---------------------------------------------------------------------------

ny = cbsa_daily[cbsa_daily["cbsa_code"] == 35620].sort_values("date_local").copy()
ny["trend"] = lowess(endog=ny["aqi"], exog=ny["date_local"], frac=0.05)[:, 1]
ny["year"] = ny["date_local"].dt.year
ny["cal_date"] = pd.to_datetime("2000-" + ny["date_local"].dt.strftime("%m-%d"))  # common leap year

years = sorted(ny["year"].unique())
fig, axs = pc.figure(len(years), 1, aspect=1.0, sharex=True, sharey=True)
for ax, year in zip(axs, years):
    d = ny[ny["year"] == year]
    ax.plot(d["cal_date"], d["aqi"], color=pc.context(), lw=0.7, label="Daily")
    ax.plot(d["cal_date"], d["trend"], color=blue, label="LOWESS trend")
    ax.set_ylim(0, 100)
    ax.set_title(str(year))
axs[len(years) // 2].set_ylabel("Daily AQI")
axs[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
pc.title(fig, "New York's air is worst in summer, every year",
         "Daily PM2.5 AQI (gray) and LOWESS trend (blue, frac = 0.05), New York metro area")
pc.note(fig, f"{AQS_NOTE}. Y-axis capped at 100; a few days run higher.")
pc.save(fig, "figures/nyc-seasonal-aqi")

# ---------------------------------------------------------------------------
# 3. ED visits by borough (notebook cell 65)
# ---------------------------------------------------------------------------

ed_daily = health.groupby(["borough", "date"])["count"].sum().reset_index(name="ed_visits")
fig, axs = pc.figure(1, 2, aspect=0.5, sharey=True)
for ax, borough in zip(axs, ["Bronx", "Queens"]):
    d = ed_daily[ed_daily["borough"] == borough]
    trend = lowess(endog=d["ed_visits"], exog=d["date"], frac=0.15, return_sorted=False)
    ax.plot(d["date"], d["ed_visits"], color=pc.context(), lw=0.7, label="Daily")
    ax.plot(d["date"], trend, color=blue, label="LOWESS trend")
    ax.set_title(f"{borough} (avg. {d['ed_visits'].mean():.0f}/day)")
axs[0].set_ylabel("ED visits per day")
axs[0].set_ylim(0, None)
axs[1].legend(loc="lower right")
pc.title(fig, "Respiratory ED visits dip in summer and peak in winter",
         "Daily asthma + respiratory ED visits, Feb 2023–Feb 2024, with LOWESS trend (frac = 0.15)")
pc.note(fig, SSD_NOTE)
pc.save(fig, "figures/ed-visits-by-borough")


# ---------------------------------------------------------------------------
# 4–5. TBR results for AQI and ED visits (notebook cells 94, 96)
# ---------------------------------------------------------------------------

def tbr_panels(m, unit, headline, name):
    fig, axs = pc.figure(3, 1, aspect=1.05, sharex=True)

    axs[0].plot(win.index, win[f"queens_{m}"], color=orange, label="Observed")
    pc.ribbon(axs[0], win.index, win[f"cf_{m}"], mask_pre(win[f"cf_{m}"] - win[f"half_{m}"]),
              mask_pre(win[f"cf_{m}"] + win[f"half_{m}"]), color=blue, label="Counterfactual")
    axs[0].legend(loc="upper left", ncols=2)
    axs[0].set_title(f"Queens {unit}: observed vs. counterfactual")

    eff = win[f"effect_{m}"]
    pc.ribbon(axs[1], win.index, eff, mask_pre(eff - win[f"half_{m}"]), mask_pre(eff + win[f"half_{m}"]),
              color=blue)
    pc.reference_line(axs[1])
    axs[1].set_title("Daily difference (observed − counterfactual)")

    cum = win[f"cum_{m}"]
    pc.ribbon(axs[2], win.index, cum, cum - win[f"cum_half_{m}"], cum + win[f"cum_half_{m}"], color=blue)
    pc.reference_line(axs[2])
    axs[2].set_title("Cumulative difference")

    for i, ax in enumerate(axs):
        pc.shade_period(ax, TEST_START, TEST_END, "Test window" if i == 2 else None)
        ax.set_ylabel(unit)
    axs[-1].xaxis.get_offset_text().set_visible(False)  # year is in the subtitle
    pc.title(fig, headline, "Time-based regression, Queens vs. a Bronx-based counterfactual, "
                            "Feb 2024, 95% intervals")
    pc.note(fig, AQS_NOTE if m == "aqi" else SSD_NOTE)
    pc.save(fig, f"figures/{name}")


final = tbr.loc[post].iloc[-1]
tbr_panels("aqi", "AQI",
           f"Queens AQI ran {final['cum_aqi']:.0f} cumulative points above its counterfactual",
           "tbr-aqi")
tbr_panels("ed", "ED visits",
           "Queens ED visits show no detectable rise over the counterfactual",
           "tbr-ed-visits")

# ---------------------------------------------------------------------------
# 6. ED visits per AQI point (notebook cell 98)
# ---------------------------------------------------------------------------

fig, axs = pc.figure(3, 1, aspect=1.05, sharex=True)
pc.ribbon(axs[0], win.index, win["cum_ed"], win["cum_ed"] - win["cum_half_ed"],
          win["cum_ed"] + win["cum_half_ed"], color=blue)
axs[0].set_title("Cumulative incremental ED visits")
pc.ribbon(axs[1], win.index, win["cum_aqi"], win["cum_aqi"] - win["cum_half_aqi"],
          win["cum_aqi"] + win["cum_half_aqi"], color=blue)
axs[1].set_title("Cumulative incremental AQI")
pc.ribbon(axs[2], win.index, win["ratio"], win["ratio"] - win["ratio_half"], win["ratio"] + win["ratio_half"],
          color=blue)
axs[2].set_title("Incremental ED visits per AQI point")
for ax, unit in zip(axs, ["ED visits", "AQI", "Ratio"]):
    pc.reference_line(ax)
    pc.shade_period(ax, TEST_START, TEST_END, "Test window" if unit == "AQI" else None)
    ax.set_ylabel(unit)
axs[-1].xaxis.get_offset_text().set_visible(False)  # year is in the subtitle

r, lo, hi = final["ratio"], final["ratio"] - final["ratio_half"], final["ratio"] + final["ratio_half"]
axs[2].annotate(f"{r:.2f} (95% CI {lo:.2f} to {hi:.2f})".replace("-", "−"), xy=(final.name, r), xytext=(-8, 18),
                textcoords="offset points", ha="right", fontsize=8, color=pc.ink("primary"),
                arrowprops=dict(arrowstyle="-", color=pc.ink("muted"), lw=0.8))
pc.title(fig, f"About {r:.2f} extra ED visits per AQI point, but the interval spans zero",
         "Ratio of cumulative incremental ED visits to cumulative incremental AQI, Queens, Feb 2024")
pc.note(fig, "Source: EPA AQS; NYC DOHMH Syndromic Surveillance. Band: delta-method 95% interval")
pc.save(fig, "figures/ed-per-aqi-ratio")
