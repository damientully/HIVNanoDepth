# ============================================================
# Sequencing depth, viral load, and ONT consensus error rate
# Publication-quality multi-panel figure (n = 68 samples)
# ============================================================
# install.packages(c("readxl","tidyverse","ggpubr","patchwork","ggcorrplot","scales"))
library(readxl)
library(tidyverse)
library(ggpubr)
library(patchwork)
library(ggcorrplot)
library(scales)

# ---------- data ----------
df <- read_excel("Viral_load_68.xlsx") %>%
  rename(
    viral_load = `Plasma_Viral _Load(copies/mL)`,
    error_rate = err_ont_weighted,
    mean_depth = mean_depth
  ) %>%
  mutate(log10_viral_load = log10(viral_load))

# ---------- statistics (reported in text/legend, not just annotated on plot) ----------
cor_depth_err <- cor.test(df$mean_depth, df$error_rate, method = "pearson")
cor_depth_err_sp <- cor.test(df$mean_depth, df$error_rate, method = "spearman")
cor_vl_err   <- cor.test(df$log10_viral_load, df$error_rate, method = "pearson")
cor_vl_depth <- cor.test(df$log10_viral_load, df$mean_depth, method = "pearson")

lm_fit <- lm(error_rate ~ log10_viral_load + mean_depth, data = df)
summary(lm_fit)

# ---------- shared theme & palette ----------
# Okabe–Ito-inspired, colorblind-safe. "Arial" falls back to "sans" if unavailable.
base_font <- "sans"
col_depth <- "#2166AC"; fill_depth <- "#CFE3FF"
col_vl    <- "#238B45"; fill_vl    <- "#D9F0D3"
col_err   <- "#54278F"; fill_err   <- "#E6D6F2"

pub_theme <- theme_classic(base_size = 10, base_family = base_font) +
  theme(
    plot.title    = element_text(face = "bold", size = 10, hjust = 0),
    axis.title    = element_text(size = 9),
    axis.text     = element_text(size = 8, colour = "black"),
    plot.caption  = element_text(size = 7.5, hjust = 0, face = "italic", colour = "grey30"),
    legend.position = "none",
    plot.margin   = margin(4, 8, 4, 4)
  )

fmt_p <- function(test) {
  p <- test$p.value
  if (p < 0.001) return("P < 0.001")
  sprintf("P = %.3f", p)
}

# ---------- Panel A–C: distributions ----------
dist_panel <- function(data, yvar, colour, fill, ylab, logy = FALSE) {
  p <- ggplot(data, aes(x = "", y = .data[[yvar]])) +
    geom_violin(fill = fill, colour = colour, alpha = 0.85, linewidth = 0.6) +
    geom_boxplot(width = 0.15, outlier.shape = NA, fill = "white", colour = colour, linewidth = 0.6) +
    geom_jitter(width = 0.06, size = 1.4, alpha = 0.7, colour = colour, stroke = 0) +
    labs(x = NULL, y = ylab) +
    pub_theme +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  if (logy) p <- p + scale_y_log10(labels = label_comma(), breaks = scales::breaks_log(n = 6))
  p
}

p1 <- dist_panel(df, "mean_depth",       col_depth, fill_depth, "Mean depth (×)", logy = TRUE)
p2 <- dist_panel(df, "log10_viral_load", col_vl,    fill_vl,    expression(log[10]~"viral load (copies/mL)"))
p3 <- dist_panel(df, "error_rate",       col_err,   fill_err,   "Weighted ONT error rate")

# ---------- Panel D–F: bivariate relationships ----------
# stat_cor() annotates the displayed bivariate association directly on the panel,
# avoiding manual caption strings that can silently go stale.
p4 <- ggplot(df, aes(x = mean_depth, y = error_rate)) +
  geom_point(size = 1.8, alpha = 0.75, colour = col_err, stroke = 0) +
  geom_smooth(method = "loess", se = TRUE, colour = col_err, fill = fill_err, linewidth = 0.8) +
  stat_cor(method = "spearman", size = 2.8, colour = "grey20", label.x.npc = 0.35) +
  scale_x_log10(labels = label_comma()) +
  labs(x = "Mean sequencing depth (×)", y = "Weighted error rate") +
  pub_theme

p5 <- ggplot(df, aes(x = log10_viral_load, y = mean_depth)) +
  geom_point(size = 1.8, alpha = 0.75, colour = col_vl, stroke = 0) +
  geom_smooth(method = "lm", se = TRUE, colour = col_vl, fill = fill_vl, linewidth = 0.8) +
  stat_cor(method = "pearson", size = 2.8, colour = "grey20") +
  scale_y_log10(labels = label_comma(), breaks = scales::breaks_log(n = 6)) +
  labs(x = expression(log[10]~"viral load (copies/mL)"), y = "Mean depth (×)") +
  pub_theme

p6 <- ggplot(df, aes(x = log10_viral_load, y = error_rate)) +
  geom_point(size = 1.8, alpha = 0.75, colour = col_err, stroke = 0) +
  geom_smooth(method = "lm", se = TRUE, colour = col_err, fill = fill_err, linewidth = 0.8) +
  stat_cor(method = "pearson", size = 2.8, colour = "grey20") +
  labs(x = expression(log[10]~"viral load (copies/mL)"), y = "Weighted error rate") +
  pub_theme

# ---------- Panel G: correlation matrix ----------
cor_df <- df %>%
  select(
    `log10 viral load`   = log10_viral_load,
    `Sequencing depth`   = mean_depth,
    `Weighted error rate`= error_rate
  )
corr <- cor(cor_df, method = "pearson")

p7 <- ggcorrplot(
  corr, type = "lower", lab = TRUE, lab_size = 3.2,
  colors = c(col_vl, "white", col_err),
  outline.color = "white", legend.title = "Pearson r"
) +
  pub_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 7.5),
    axis.text.y = element_text(size = 7.5),
    legend.position = "right",
    legend.key.height = unit(0.35, "in")
  )

# ---------- assemble ----------
final_fig <- (p1 | p2 | p3 | p4) / (p5 | p6 | p7) +
  plot_annotation(
    title = "Relationship between plasma viral load, sequencing depth, and consensus error rate",
    subtitle = paste0("Oxford Nanopore whole-genome sequencing, n = ", nrow(df), " samples"),
    tag_levels = "A",
    theme = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10, colour = "grey30")
    )
  )

# ---------- save (vector PDF for submission; high-res PNG for preview/PPT) ----------
ggsave("sequencing_depth_viral_load_error_figure.pdf", final_fig, width = 12, height = 7.2, units = "in", device = cairo_pdf)
ggsave("sequencing_depth_viral_load_error_figure.png", final_fig, width = 12, height = 7.2, units = "in", dpi = 600)
