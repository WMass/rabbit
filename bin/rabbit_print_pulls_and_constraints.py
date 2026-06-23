#!/usr/bin/env python3

import numpy as np

from rabbit import io_tools, parsing

sort_choices = []
sort_choices_abs = [f"abs {s}" for s in sort_choices]


def make_parser():
    parser = parsing.print_parser()
    parser.add_argument(
        "-s",
        "--sort",
        type=str,
        default=None,
        choices=[
            "label",
            "pull",
            "constraint",
            "pull prefit",
            "constraint prefit",
            "abs pull",
            "abs pull prefit",
        ],
        help="Sort parameters according to criteria, do not sort by default",
    )
    parser.add_argument(
        "--reverse-sort",
        default=False,
        action="store_true",
        help="Reverse the sorting",
    )
    parser.add_argument(
        "--asym",
        default=False,
        action="store_true",
        help="Print asymmetric constraints from contour scans",
    )
    parser.add_argument(
        "--keepNuisances",
        type=str,
        default=None,
        help="Match nuisances by regular expression",
    )
    parser.add_argument(
        "--excludeNuisances",
        type=str,
        default=None,
        help="Exclude nuisances by regular expression",
    )
    parser.add_argument(
        "--noPrefit",
        default=False,
        action="store_true",
        help="Suppress the prefit pull and constraint columns from the printout",
    )
    return parser


def main():
    args = make_parser().parse_args()
    fitresult = io_tools.get_fitresult(args.infile, args.result)

    labels, pulls, constraints = io_tools.get_pulls_and_constraints(
        fitresult,
        keep_nuisances=args.keepNuisances,
        exclude_nuisances=args.excludeNuisances,
    )
    labels, pulls_prefit, constraints_prefit = io_tools.get_pulls_and_constraints(
        fitresult,
        prefit=True,
        keep_nuisances=args.keepNuisances,
        exclude_nuisances=args.excludeNuisances,
    )

    if args.asym:
        _0, _1, constraints_asym = io_tools.get_pulls_and_constraints(
            fitresult,
            keep_nuisances=args.keepNuisances,
            exclude_nuisances=args.excludeNuisances,
            asym=True,
        )

    if args.sort is not None:
        if args.sort.startswith("abs"):
            f = lambda x: abs(x)
            sort = args.sort.replace("abs ", "")
        else:
            f = lambda x: x
            sort = args.sort

        if sort == "label":
            order = np.argsort(labels)
        elif sort == "pull":
            order = np.argsort(f(pulls))
        elif sort == "constraint":
            order = np.argsort(f(constraints))
        elif sort == "pull prefit":
            order = np.argsort(f(pulls_prefit))
        elif sort == "constraint prefit":
            order = np.argsort(f(constraints_prefit))

        if args.reverse_sort:
            order = order[::-1]

        labels = labels[order]
        pulls = pulls[order]
        constraints = constraints[order]
        pulls_prefit = pulls_prefit[order]
        constraints_prefit = constraints_prefit[order]

        if args.asym:
            constraints_asym = constraints_asym[order]

    nround = 5
    prefit_header = (
        "" if args.noPrefit else f" ({'pull prefit':>11} +/- {'constraint prefit':>17})"
    )
    if args.asym:
        header = f"   {'Parameter':<30} {'pull':>6} +/- {'constraint':>10} + {'up':>10} - {'down':>10}{prefit_header}"
        print(header)
        print("   " + "-" * (len(header) - 3))
        print(
            "\n".join(
                [
                    f"   {l:<30} {round(p, nround):>6} +/- {round(c, nround):>10} + {round(c_asym[0], nround):>10} - {round(c_asym[1], nround):>10}"
                    + (
                        ""
                        if args.noPrefit
                        else f" ({round(pp, nround):>11} +/- {round(pc, nround):>17})"
                    )
                    for l, p, c, c_asym, pp, pc in zip(
                        labels,
                        pulls,
                        constraints,
                        constraints_asym,
                        pulls_prefit,
                        constraints_prefit,
                    )
                ]
            )
        )
    else:
        header = (
            f"   {'Parameter':<30} {'pull':>6} +/- {'constraint':>10}{prefit_header}"
        )
        print(header)
        print("   " + "-" * (len(header) - 3))
        print(
            "\n".join(
                [
                    f"   {l:<30} {round(p, nround):>6} +/- {round(c, nround):>10}"
                    + (
                        ""
                        if args.noPrefit
                        else f" ({round(pp, nround):>11} +/- {round(pc, nround):>17})"
                    )
                    for l, p, c, pp, pc in zip(
                        labels, pulls, constraints, pulls_prefit, constraints_prefit
                    )
                ]
            )
        )


if __name__ == "__main__":
    main()
