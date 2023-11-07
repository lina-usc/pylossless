import numpy as np


def _get_ics(df, ic_type):
    if not df.empty:
        return df[df["ic_type"] == ic_type]["component"].tolist()
    return None


def _sum_flagged_times(raw, flags):
    """Sum the total time flagged for various flags like noisy etc."""
    flag_dict = {}
    for flag in flags:
        flag_dict[flag] = []
        if raw:
            inds = np.where(raw.annotations.description == flag)[0]
            if len(inds):
                flag_dict[flag] = np.sum(raw.annotations.duration[inds])
    return flag_dict


def _create_html_details(title, data, times=False):
    html_details = f"<details><summary><strong>{title}</strong></summary>"
    html_details += "<table>"
    for key, value in data.items():
        if times:  # special format for flagged times
            value = f"{value:.2f} s" if value else value
            html_details += f"<tr><td>{key}</td><td>{value} seconds</td></tr>"
        else:  # Channels, ICs
            html_details += f"<tr><td>{key}</td><td>{value}</td></tr>"
    html_details += "</table></details>"
    return html_details
