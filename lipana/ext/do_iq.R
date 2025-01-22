# This script performs MaxLFQ estimation via iq on the input file
# The input file path is required as the first argument
# An optional second argument specifies whether to perform normalization (default: FALSE)
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
    stop("Error: Input file path is required for MaxLFQ estimation via iq", call. = FALSE)
}

infile <- args[1]
if (!file.exists(infile)) {
    stop(paste("Input file", infile, "does not exist"), call. = FALSE)
}

norm_flag <- if (length(args) >= 2) as.logical(args[2]) else FALSE
message(sprintf("MaxLFQ estimation via iq for %s", infile))

indata <- iq::fast_read(
    infile,
    sample_id = "SampleIds",
    primary_id = "PrimaryIds",
    secondary_id = "AggregationIds",
    intensity_col = "BaseQuant",
    annotation_col = NULL,
    filter_string_equal = NULL,
    filter_string_not_equal = NULL,
    filter_double_less = NULL,
    filter_double_greater = NULL,
    intensity_col_sep = NULL,
    intensity_col_id = NULL,
    na_string = "0"
)

norm_indata <- iq::fast_preprocess(
    indata$quant_table,
    median_normalization = norm_flag,
    log2_intensity_cutoff = 0,
    pdf_out = NULL,
    show_boxplot = FALSE
)

result <- iq::fast_MaxLFQ(norm_indata)

final_report <- data.frame(
    PrimaryIds = indata$protein,
    result$estimate
)
colnames(final_report)[-1] <- indata$sample
outfile <- sprintf('%s-iq_output.txt', infile)
message(sprintf("Output to %s", outfile))
write.table(final_report, outfile, sep = "\t", row.names = FALSE)
