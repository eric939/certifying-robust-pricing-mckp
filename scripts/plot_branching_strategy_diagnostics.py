#!/usr/bin/env python3
"""Plot tight-capacity branching-strategy diagnostics using base R."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_INPUT = Path("results/branching_diagnostics/branching_diagnostics.csv")
DEFAULT_OUTPUT = Path("paper_versions/v2/figures/branching_diagnostics")


def _has_rows(input_path: Path) -> bool:
    if not input_path.exists():
        return False
    with input_path.open(newline="", encoding="utf-8") as f:
        return bool(list(csv.DictReader(f)))


def _r_script() -> str:
    return r'''
args <- commandArgs(trailingOnly=TRUE)
input <- args[[1]]
outdir <- args[[2]]
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
d <- read.csv(input, stringsAsFactors=FALSE)
if (nrow(d) == 0) quit(status=0)
d$rule_label <- gsub("_", "\n", d$branch_rule)

save_plot <- function(name, expr, width=8.5, height=4.8) {
  pdf(file.path(outdir, paste0(name, ".pdf")), width=width, height=height)
  par(mar=c(8, 4.5, 2, 1), cex.axis=0.82)
  expr()
  dev.off()
  png(file.path(outdir, paste0(name, ".png")), width=1400, height=850, res=180)
  par(mar=c(8, 4.5, 2, 1), cex.axis=0.82)
  expr()
  dev.off()
}

box_by_rule <- function(field, ylab, name, log_axis=FALSE) {
  vals <- d[[field]]
  ok <- is.finite(vals)
  if (!any(ok)) return()
  f <- factor(d$rule_label[ok], levels=unique(d$rule_label[ok]))
  y <- vals[ok]
  save_plot(name, function() {
    boxplot(y ~ f, ylab=ylab, xlab="", las=2, col="gray85", border="black",
            outline=TRUE, log=if (log_axis) "y" else "")
    grid(nx=NA, ny=NULL, col="gray85")
  })
}

box_by_rule("relative_gap", "Final relative gap", "branching_gap_by_rule")
box_by_rule("runtime_seconds", "Runtime (s)", "branching_runtime_by_rule")
box_by_rule("nodes_explored", "Nodes explored", "branching_nodes_by_rule", TRUE)

rules <- unique(d$branch_rule)
if (length(rules) > 0) {
  prunes <- rbind(
    bound=sapply(rules, function(r) sum(d$nodes_pruned_bound[d$branch_rule == r], na.rm=TRUE)),
    infeasible=sapply(rules, function(r) sum(d$nodes_pruned_infeasible[d$branch_rule == r], na.rm=TRUE)),
    cutoff=sapply(rules, function(r) sum(d$nodes_pruned_cutoff[d$branch_rule == r], na.rm=TRUE))
  )
  colnames(prunes) <- gsub("_", "\n", rules)
  save_plot("branching_bound_prunes_by_rule", function() {
    barplot(prunes + 1, beside=FALSE, log="y", ylab="Pruned nodes + 1",
            col=c("gray25", "gray60", "gray85"), border="black", las=2)
    legend("topleft", legend=rownames(prunes), fill=c("gray25", "gray60", "gray85"),
           bty="n", horiz=TRUE, cex=0.85)
    grid(nx=NA, ny=NULL, col="gray85")
  })
}

ok <- is.finite(d$strong_branching_time) & is.finite(d$relative_gap)
if (any(ok) && max(d$strong_branching_time[ok], na.rm=TRUE) > 0) {
  save_plot("strong_branching_tradeoff", function() {
    plot(d$strong_branching_time[ok], d$relative_gap[ok], pch=1, col="black",
         xlab="Strong-branching time (s)", ylab="Final relative gap")
    text(d$strong_branching_time[ok], d$relative_gap[ok], labels=d$branch_rule[ok],
         pos=4, cex=0.62)
    grid(col="gray85")
  }, width=7.6, height=4.8)
}
'''


def plot(input_path: Path, output_dir: Path) -> None:
    if not _has_rows(input_path):
        return
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required for this lightweight plotting helper")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as f:
        f.write(_r_script())
        script_path = Path(f.name)
    try:
        subprocess.run([rscript, str(script_path), str(input_path), str(output_dir)], check=True)
    finally:
        script_path.unlink(missing_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plot(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
