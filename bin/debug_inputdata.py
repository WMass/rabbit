#!/usr/bin/env python
import argparse

import hist
import numpy as np
from wums import logging

from combinetf2 import debugdata, inputdata

logger = None


def debug_input_data(input_file, output_dir=None, verbose=False, channels=None):
    """
    Debug input data file and report potential issues
    """
    logger.info(f"Debugging input file: {input_file}")

    # Load the input data
    indata = inputdata.FitInputData(input_file)

    try:
        debug_data = debugdata.FitDebugData(indata)
        logger.info("✓ Successfully created FitDebugData object")
    except Exception as e:
        logger.info(f"✗ Failed to create FitDebugData: {str(e)}")
        return

    if channels is None:
        channels = [k for k in debug_data.nominal_hists.keys()]

    # Check for issues
    issues_found = 0

    # 1. Check for channels with no data observations
    logger.info("\nChecking channels with data observations:")
    channels_with_data = list(
        [k for k in debug_data.data_obs_hists.keys() if k in channels]
    )
    channels_without_data = [
        ch
        for ch in debug_data.indata.channel_info.keys()
        if ch not in channels_with_data
        and not indata.channel_info[ch]["masked"]
        and ch in channels
    ]

    if channels_without_data:
        issues_found += 1
        logger.info(
            f"✗ Found {len(channels_without_data)} channels without data observations:"
        )
        for ch in channels_without_data:
            logger.info(f"  - {ch}")
    else:
        logger.info("✓ All channels have data observations")

    # 2. Check for empty channels in data
    empty_data_channels = []
    for channel, hist_obj in debug_data.data_obs_hists.items():
        if channel not in channels:
            continue
        if np.sum(hist_obj.values()) == 0:
            empty_data_channels.append(channel)

    if empty_data_channels:
        issues_found += 1
        logger.info(
            f"✗ Found {len(empty_data_channels)} channels with empty data observations:"
        )
        for ch in empty_data_channels:
            logger.info(f"  - {ch}")
    else:
        logger.info("✓ All channels have non-empty data observations")

    # 2.1 Check for bins with low number of data events
    threshold = 10
    for channel, hist_obj in debug_data.data_obs_hists.items():
        if channel not in channels:
            continue
        if np.sum(hist_obj.values() < threshold):
            idxs = np.where(hist_obj.values() < threshold)
            low_data_bins = idxs

            issues_found += 1
            logger.info(
                f"✗ Found {len(low_data_bins)} bins in channel {channel} with less than {threshold} data observations:"
            )
            for idxs in zip(*low_data_bins):
                bins_str = ", ".join(
                    [f"{name}: {idx}" for idx, name in zip(idxs, hist_obj.axes.name)]
                )
                logger.info(f"  - {hist_obj.values()[idxs]} events in bin {bins_str}")
        else:
            logger.info(
                f"✓ All bins in channel {channel} have more than {threshold} data observations"
            )

    # 2.2 Check for bins with low number of prediction
    threshold_yield = 1
    for channel, hist_obj in debug_data.nominal_hists.items():
        if channel not in channels:
            continue
        hist_proc = hist_obj[{"processes": hist.sum}]

        if np.sum(hist_proc.values() < threshold_yield):
            idxs = np.where(hist_proc.values() < threshold_yield)
            low_yield_bins = idxs

            issues_found += 1
            logger.info(
                f"✗ Found {len(low_yield_bins[0])} bins in channel {channel} with less than {threshold_yield} prediction:"
            )
            for idxs in zip(*low_yield_bins):
                bins_str = ", ".join(
                    [f"{name}: {idx}" for idx, name in zip(idxs, hist_proc.axes.name)]
                )
                logger.info(f"  - {hist_proc.values()[idxs]} events in bin {bins_str}")
        else:
            logger.info(
                f"✓ All bins in channel {channel} have more than {threshold_yield} prediction"
            )

    # 2.3 Check for bins with high relative stat uncertainty in prediction
    threshold_rel_unc = 1
    for channel, hist_obj in debug_data.nominal_hists.items():
        if channel not in channels:
            continue
        hist_proc = hist_obj[{"processes": hist.sum}]
        values = hist_proc.variances() ** 0.5 / hist_proc.values()
        if np.sum(values > threshold_rel_unc):
            idxs = np.where(values > threshold_rel_unc)
            high_unc_bins = idxs

            issues_found += 1
            logger.info(
                f"✗ Found {len(high_unc_bins[0])} bins in channel {channel} with relative uncertainty larger than {threshold_rel_unc*100}% in prediction"
            )
            for idxs in zip(*high_unc_bins):
                bins_str = ", ".join(
                    [f"{name}: {idx}" for idx, name in zip(idxs, hist_proc.axes.name)]
                )
                logger.info(
                    f"  - {values[idxs]} relative uncertainty in bin {bins_str}"
                )
        else:
            logger.info(
                f"✓ All bins in channel {channel} have less than {threshold_rel_unc*100}% relative uncertainty in prediction"
            )

    # 3. Check for processes with zero normalization
    zero_norm_procs = {}
    for channel, hist_obj in debug_data.nominal_hists.items():
        if channel not in channels:
            continue
        proc_sums = np.sum(hist_obj.values(), axis=tuple(range(hist_obj.ndim - 1)))
        zero_procs = [
            debug_data.indata.procs[i] for i, val in enumerate(proc_sums) if val == 0
        ]
        if zero_procs:
            zero_norm_procs[channel] = zero_procs

    if zero_norm_procs:
        issues_found += 1
        logger.info("✗ Found processes with zero normalization:")
        for channel, procs in zero_norm_procs.items():
            logger.info(
                f"  - Channel {channel}: {', '.join(np.array(procs, dtype=str))}"
            )
    else:
        logger.info("✓ All processes have non-zero normalization")

    # 4. Check for systematics with only zeros
    all_systs = list(debug_data.axis_systs)
    nonzero_systs = debug_data.nonzeroSysts()
    zero_systs = [syst for syst in all_systs if syst not in nonzero_systs]

    if zero_systs:
        issues_found += 1
        logger.info(f"✗ Found {len(zero_systs)} systematics with only zeros:")
        for syst in zero_systs:
            logger.info(f"  - {syst}")
    else:
        logger.info("✓ All systematics are nonzero")

    # 5. Check for processes with no systematic variations
    procs_without_systs = []
    for proc in debug_data.indata.procs:
        channels_with_syst = debug_data.channelsForNonzeroSysts(
            procs=[proc.decode("utf-8")]
        )
        if not channels_with_syst:
            procs_without_systs.append(proc)

    if procs_without_systs:
        issues_found += 1
        logger.info(
            f"✗ Found {len(procs_without_systs)} processes with no systematic variations:"
        )
        for proc in procs_without_systs:
            logger.info(f"  - {proc}")
    else:
        logger.info("✓ All processes have systematic variations")

    # 6. Check for extreme systematic variations
    extreme_variations = {}
    threshold = 2.0  # Variation more than 100% up or down

    for channel, syst_hist in debug_data.syst_hists.items():
        if channel not in channels:
            continue
        nom_hist = debug_data.nominal_hists[channel]
        # Calculate ratio of systematic variation to nominal
        down_values = syst_hist[{"DownUp": "Down"}].values()
        up_values = syst_hist[{"DownUp": "Up"}].values()
        nom_values = nom_hist.values()[..., None]

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            down_ratio = np.where(nom_values > 0, down_values / nom_values, np.nan)
            up_ratio = np.where(nom_values > 0, up_values / nom_values, np.nan)

        # Find extreme variations
        for isyst, syst in enumerate(debug_data.indata.systs):
            for iproc, proc in enumerate(debug_data.indata.procs):
                down_extreme = np.nanmin(down_ratio[..., iproc, isyst])
                up_extreme = np.nanmax(up_ratio[..., iproc, isyst])

                if down_extreme < 1 / threshold or up_extreme > threshold:
                    if channel not in extreme_variations:
                        extreme_variations[channel] = []
                    extreme_variations[channel].append(
                        (proc, syst, down_extreme, up_extreme)
                    )

    if extreme_variations:
        issues_found += 1
        logger.info("✗ Found extreme systematic variations (>100%):")
        for channel, variations in extreme_variations.items():
            logger.info(f"  Channel: {channel}")
            for proc, syst, down, up in variations:
                logger.info(f"    - Process: {proc}, Systematic: {syst}")
                logger.info(f"      Down: {down:.2f}x, Up: {up:.2f}x")
    else:
        logger.info("✓ No extreme systematic variations found")

    # Summary
    if issues_found == 0:
        logger.info("\n✓ No issues found in the input data!")
    else:
        logger.info(f"\n✗ Found {issues_found} potential issues in the input data")

    return issues_found == 0


def main():
    parser = argparse.ArgumentParser(description="Debug input data for fitting")
    parser.add_argument("inputFile", help="Path to input data file")
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger", action="store_true", help="Do not use logging with colors"
    )
    parser.add_argument(
        "--channels",
        type=str,
        help="Only check specified channels",
        default=None,
        nargs="+",
    )
    args = parser.parse_args()

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    debug_input_data(args.inputFile, args.verbose, channels=args.channels)


if __name__ == "__main__":
    main()
