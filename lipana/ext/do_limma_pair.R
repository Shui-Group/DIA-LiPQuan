# This script is used to perform limma on the input file
# This will do pairwise comparison between conditions, for group comparison, see do_limma_group.R
# This script requires at least 3 arguments:
# 1. A quantification matrix file, with rows as entries and columns as runs, and a column to annotate entry name in each row
# 2. A annotation file, with at least two columns "run" and "condition" (can directly use the experiment setting file used in lipana); or a string in format like "run1::c1;;run2::c1;;run3::c2;;run4::c2" to specify the condition for each run
# 3. A string to specify the column name for the entry column in the input matrix file, such as "precursor", by default it is "Entry"
# 4. An optional string to specify required condition pairs, separated by ";;" and can be a mix of the following formats:
#   a. a single condition to be the base condition, like "cond1;;cond2" will be flatten as "cond2//cond1;;cond3//cond1;;cond1//cond2;;cond3//cond2"
#   b. condition pairs separated by "//", like "cond1//cond2;;cond3//cond2"
#   c. if this argument is not provided, will compare all available pairs
#   (a and b can be mixed, like "cond1//cond3;;cond2" will be flatten as "cond1//cond3;;cond1//cond2;;cond3//cond2")
# Note: For each pair, each condition requires at least 2 runs with quantification values

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop("Error: At least two inputs are required: an input matrix file and an annotation file", call. = FALSE)
}

infile <- args[1]
if (!file.exists(infile)) {
    stop(paste("Input file", infile, "does not exist"), call. = FALSE)
}
message(sprintf("Quantification file: %s", infile))

inanno <- args[2]
if (is.character(inanno)) {
    if (file.exists(inanno)) {
        annotation <- read.delim(inanno)
    } else if (grepl("::", inanno)) {
        annotation <- as.data.frame(do.call(rbind, strsplit(unlist(strsplit(inanno, ";;")), "::")))
        colnames(annotation) <- c("run", "condition")
    } else {
        stop("Error: Annotation file or string is required", call. = FALSE)
    }
} else {
    stop("Error: Annotation file or string is required", call. = FALSE)
}
all_conditions <- unique(annotation$condition)

entry_col <- if (length(args) >= 3) args[3] else "Entry"

raw_pairs <- if (length(args) >= 4) args[4] else NULL
comp_pairs <- list()
if (is.null(raw_pairs) || raw_pairs == "None") {
    comp_pairs <- combn(all_conditions, 2, simplify = FALSE)
} else {
    for (pair in strsplit(raw_pairs, ";;")[[1]]) {
        if (grepl("//", pair)) {
            comp_pairs[[length(comp_pairs) + 1]] <- strsplit(pair, "//")[[1]]
        } else {
            for (cond in all_conditions){
                if (pair != cond){
                    comp_pairs[[length(comp_pairs) + 1]] <- c(cond, pair)
                }
            }
        }
    }
}

if (grepl("\\.parquet$", infile)) {
    indata <- as.data.frame(arrow::read_parquet(infile))
    indata[] <- lapply(indata, function(col) {
        if (is.character(col)) col[col == "NA"] <- NA
        return(col)
    })
} else {
    indata <- read.delim(infile, na.strings = "NA", check.names = FALSE)
}
numeric_columns <- intersect(annotation$run, colnames(indata))
indata[numeric_columns] <- lapply(indata[numeric_columns], as.numeric)

avail_conditions <- unique(annotation[annotation$run %in% numeric_columns, "condition"])
avail_comp_pairs <- Filter(function(pair) all(pair %in% avail_conditions), comp_pairs)
message(sprintf("All defined comparison pairs: %s", comp_pairs))
message(sprintf("Available comparison pairs: %s", avail_comp_pairs))

output <- list()
for (pair in avail_comp_pairs) {
    cond1 <- pair[1]
    cond2 <- pair[2]
    message(sprintf("Comparing %s vs %s", cond1, cond2))

    cond1_runs <- annotation[annotation$condition == cond1, "run"]
    cond2_runs <- annotation[annotation$condition == cond2, "run"]
    all_runs <- c(cond1_runs, cond2_runs)

    design <- cbind(
        base = 1,
        c1vsc2 = c(rep(1, length(cond1_runs)), rep(0, length(cond2_runs)))
    )
    rownames(design) <- all_runs

    sub_indata <- indata[
        rowSums(!is.na(indata[cond1_runs])) >= 2 & rowSums(!is.na(indata[cond2_runs])) >= 2, 
        c(all_runs, entry_col)
    ]
    if (nrow(sub_indata) == 0) {
        message("No row left after requiring at least 2 detections for each condition. Skip this pair.")
        next
    }

    input_matrix <- as.matrix(sub_indata[, all_runs])
    rownames(input_matrix) <- sub_indata[[entry_col]]

    fit <- limma::lmFit(input_matrix, design = design)
    fit <- limma::eBayes(fit, proportion = 0.01)

    stats_table <- limma::topTable(
        fit,
        coef = 2,
        number = nrow(fit),
        genelist = sub_indata[[entry_col]],
        adjust.method = "BH",
        p.value = 1
    )

    if (nrow(stats_table) > 0) {
        stats_table["pair"] <- sprintf("%s_vs_%s", cond1, cond2)
        output[[length(output) + 1]] <- stats_table
    }
}

output <- do.call(rbind, output)

outfile <- sprintf("%s-limma_output.txt", infile)
message(sprintf("Output to %s", outfile))
write.table(output, outfile, sep = "\t", row.names = FALSE)
