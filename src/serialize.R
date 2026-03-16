args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || !nzchar(args[1])) {
    stop("WORKDIR argument is required.")
}
workdir <- args[1]

# Create regimens multi-object: regimens, concepts
regimens <- read.delim(file.path(workdir, "regimens.tsv"),
                                             stringsAsFactors = FALSE)
concepts <- read.delim(file.path(workdir, "concepts.tsv"),
                                            stringsAsFactors = FALSE)
save(regimens, concepts, file = file.path(workdir, "regimens.rda"))

validdrugs <- read.delim(file.path(workdir, "validdrugs.tsv"),
                                                 stringsAsFactors = FALSE)
save(validdrugs, file = file.path(workdir, "validdrugs.rda"))

regimengroups <- read.delim(file.path(workdir, "regimengroups.tsv"),
                                                        stringsAsFactors = FALSE)
save(regimengroups, file = file.path(workdir, "regimengroups.rda"))