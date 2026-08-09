#!/usr/bin/env python3
"""Generate final Path A publication experiment plots.

This helper uses base R because the local Matplotlib font cache can stall for
minutes in the current desktop environment.  The output is deterministic,
grayscale-safe, and generated only from the benchmark CSV.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence


def r_script() -> str:
    return r'''
args <- commandArgs(trailingOnly=TRUE)
input <- args[[1]]
outdir <- args[[2]]
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
d <- read.csv(input, stringsAsFactors=FALSE)
families <- c("economic", "many_theta", "hull_compression", "adversarial")
family_lab <- c(economic="Economic", many_theta="Many θ", hull_compression="Hull compression", adversarial="Low compression")
method_lab <- c(hullround="HullRound", global_bnb_cached_cutoff_ordered="Exact θ-B&B", scip="SCIP", highs="HiGHS")
inst_key <- function(x) paste(x$family, x$n, x$m, x$gamma, x$gamma_mode, x$seed, sep="|")

save_plot <- function(name, expr, width=7.2, height=4.4) {
  pdf(file.path(outdir, paste0(name, ".pdf")), width=width, height=height)
  par(mar=c(6.5, 4.5, 2, 1), cex.axis=0.86)
  expr()
  dev.off()
  png(file.path(outdir, paste0(name, ".png")), width=1400, height=850, res=190)
  par(mar=c(6.5, 4.5, 2, 1), cex.axis=0.86)
  expr()
  dev.off()
}

hr <- d[d$method == "hullround",]
ex <- d[d$method == "global_bnb_cached_cutoff_ordered" & d$status == "optimal",]
hr_map <- setNames(hr$objective, inst_key(hr))
gaps <- data.frame()
if (nrow(ex) > 0) {
  for (i in seq_len(nrow(ex))) {
    k <- inst_key(ex[i,])
    if (k %in% names(hr_map) && is.finite(hr_map[[k]]) && ex$objective[i] > 0) {
      gap <- max(0, (ex$objective[i] - hr_map[[k]]) / ex$objective[i])
      gaps <- rbind(gaps, data.frame(family=ex$family[i], gamma_mode=ex$gamma_mode[i], gap_bp=10000*gap))
    }
  }
}

if (nrow(gaps) > 0) {
  gaps$family <- factor(gaps$family, levels=families, labels=family_lab[families])
  save_plot("final_hullround_gap_by_family", function() {
    boxplot(gap_bp ~ family, data=gaps, col="gray85", border="black",
            ylab="HullRound gap to exact optimum (basis points)", xlab="", las=2)
    stripchart(gap_bp ~ family, data=gaps, vertical=TRUE, method="jitter", pch=1,
               col="black", add=TRUE)
    grid(nx=NA, ny=NULL, col="gray85")
  })
  save_plot("final_gap_histogram", function() {
    hist(gaps$gap_bp, breaks=12, col="gray85", border="black",
         xlab="HullRound gap to exact optimum (basis points)", main="")
    grid(nx=NA, ny=NULL, col="gray85")
  }, width=6.4, height=4.2)
}

runtime <- aggregate(runtime_seconds ~ method + n, data=d, FUN=median)
runtime <- runtime[runtime$method %in% names(method_lab),]
if (nrow(runtime) > 0) {
  save_plot("final_runtime_tradeoff", function() {
    plot(NA, xlim=range(runtime$n), ylim=range(runtime$runtime_seconds[runtime$runtime_seconds>0]),
         log="y", xlab="Number of items n", ylab="Median runtime (seconds, log scale)")
    ltys <- c(hullround=1, global_bnb_cached_cutoff_ordered=2, scip=3, highs=4)
    pchs <- c(hullround=1, global_bnb_cached_cutoff_ordered=2, scip=5, highs=4)
    for (m in names(method_lab)) {
      r <- runtime[runtime$method == m,]
      if (nrow(r) > 0) lines(r$n, r$runtime_seconds, type="b", lty=ltys[[m]], pch=pchs[[m]], col="black")
    }
    legend("topleft", legend=method_lab[names(method_lab)], lty=c(1,2,3,4), pch=c(1,2,5,4),
           col="black", bty="n", cex=0.82)
    grid(col="gray85")
  })
}

cert <- d[d$method %in% c("global_bnb_cached_cutoff_ordered", "scip", "highs"),]
if (nrow(cert) > 0) {
  cert$certified <- cert$status == "optimal"
  tab <- tapply(cert$certified, list(cert$family, cert$method), sum)
  tab[is.na(tab)] <- 0
  tab <- tab[families[families %in% rownames(tab)],,drop=FALSE]
  rownames(tab) <- family_lab[rownames(tab)]
  colnames(tab) <- method_lab[colnames(tab)]
  save_plot("final_certification_by_family", function() {
    barplot(t(tab), beside=TRUE, col=c("gray20","gray60","gray85"), border="black",
            ylab="Certified optimal rows", las=2)
    legend("topright", legend=colnames(tab), fill=c("gray20","gray60","gray85"),
           bty="n", cex=0.82)
    grid(nx=NA, ny=NULL, col="gray85")
  }, width=7.8, height=4.8)
}
'''


def plot(input_path: Path, output_dir: Path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required for this plotting helper")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as f:
        f.write(r_script())
        script_path = Path(f.name)
    try:
        subprocess.run([rscript, str(script_path), str(input_path), str(output_dir)], check=True)
    finally:
        script_path.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    plot(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
