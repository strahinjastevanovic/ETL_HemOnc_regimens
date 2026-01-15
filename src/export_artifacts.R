save_regimen_bundle <- function(workdir) {
  regimens <- read.delim(file.path(workdir, "regimens.tsv"), stringsAsFactors = FALSE)
  drugs <- read.delim(file.path(workdir, "regimens_drugs.tsv"), stringsAsFactors = FALSE)

  save(
    regimens,
    drugs,
    file = file.path(workdir, "regimens.rda")
  )

  invisible(TRUE)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 1) {
    stop("Usage: Rscript save_regimen_bundle.R <WORKDIR>")
  }
  
  save_regimen_bundle(args[[1]])
  
  validdrugs <- read.delim("${WORKDIR}/validdrugs.tsv", stringsAsFactors = FALSE)
  save(validdrugs, file = "${WORKDIR}/validdrugs.rda")
  
  regimengroups <- read.delim("${WORKDIR}/regimengroups.tsv", stringsAsFactors = FALSE)
  save(regimengroups, file = "${WORKDIR}/regimengroups.rda")
  
}
