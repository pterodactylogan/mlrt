library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

# Helper function to add column adding the data types 
add_type <- function(df, type){
  new_df <- df %>% mutate(data_type = type)
  return(new_df)
}


# Load in and clean and fix all the neural data 
load_nn <- function(path){
  # Load in and label all the nn results 
  neural_os <- add_type(
    read.csv(file.path(path, "onlyshort_evals.csv")),
    "OS")
  neural_ps <- add_type(
    read.csv(file.path(path, "plusshort_evals.csv")),
    "PS")
  neural_ol <- add_type(
    read.csv(file.path(path, "standard_evals.csv")),
    "OL")
  # Clean it so the column names are the same as ff 
  neural_df = rbind(neural_os, neural_ps, neural_ol) %>%
    mutate(recall = tp / (tp+fn)) %>%
    filter(alph < 64) %>%
    select(data_type,
           alphabet_size = alph, 
           tier_size = tier, 
           language_class = class, 
           factor_width = k, 
           threshold = j, 
           index = i, 
           split = test_type,
           accuracy, 
           precision, 
           recall, 
           f1 = fscore,
           brier_score = brier,
           model = network_type,
           train_size = train_set_size
    ) 
  return(neural_df)
}

# Load in and clean and fix all the FF data 
load_ff <- function(path){
  ff_os <- add_type(
    read.csv(file.path(path, "models-os/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")),
    "OS") %>% mutate(train_size = "Small")
  for (size in c("Small", "Mid")){
    ps_size <- add_type(
      read.csv(paste0("../FlexFringe/FlexFringe/models-ps-", tolower(size), "/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")),
      "PS")
    ol_size <- add_type(
      read.csv(paste0("../FlexFringe/FlexFringe/models-reg-", tolower(size), "/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")),
      "OL")
    size_df <- rbind(ps_size, ol_size) %>%
      mutate(train_size = size)
    ff_os <- rbind(ff_os, size_df)
  }
  ff_df <- ff_os %>% mutate(split = str_remove(split, "Test")) %>%
    filter(!split %in% c("Train", "Dev")) %>%
    mutate(model = "FF") %>%
    filter(alphabet_size < 64) %>%
    select(-data_size, -ini, -model_path, -data_path, -last_modified)
  return(ff_df)
}

# Load in the data
all_neural <- load_nn("../neural")
all_ff <- load_ff("../FlexFringe/FlexFringe")
everything <- rbind(all_neural, all_ff)

# write.csv(everything, "everything.csv", row.names = FALSE)

# Make Table 1 in the paper 
table1 <- everything %>%
  group_by(model, data_type, train_size) %>%
  summarize(meanf1 = round(mean(f1), 3))

# Make Table 2 in the paper 
table2 <- everything %>%
  filter(data_type == "OS") %>%
  group_by(model, split) %>% 
  summarize(meanf1 = round(mean(f1), 3))

# Make Table 3 in the paper 
table3 <- everything %>% 
  filter(data_type == "PS" & train_size == "Small") %>%
  group_by(model, split) %>% 
  summarize(meanf1 = round(mean(f1), 3))

# Make Table 4 in the paper 
table4 <- everything %>% 
  filter(data_type == "OL" & train_size == "Small") %>% 
  group_by(model, split) %>% 
  summarize(meanf1 = round(mean(f1), 3))


# Get the differences in performance by language and make boxplots
diffs <- everything %>%
  filter(train_size == "Small") %>%
  select(-c(brier_score, recall, precision, accuracy)) %>%
  pivot_wider(names_from = data_type, values_from = f1) %>%
  drop_na()

# PS vs. OS 
diffs %>% ggplot(aes(x = model, y = PS - OS, fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Difference in Accuracy") + 
  xlab("") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1), 
        legend.position = "none"
  ) + 
  ggtitle("Difference in Accuracy from Small Plus Short to Only Short by Model")
ggsave("figs/acc-diff-ps-os.pdf", width=6, height=4)

# PS vs. OL
diffs %>% ggplot(aes(x = model, y = PS - OL, fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Difference in Accuracy") + 
  xlab("") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1), 
        legend.position = "none"
  ) + 
  ggtitle("Difference in Accuracy from Small Plus Short to Only Small by Model")
ggsave("figs/acc-diff-ps-ol.pdf", width=6, height=4)

# Combined boxplot
diffs$'PS-OS' <- diffs$PS - diffs$OS
diffs$'PS-OL' <- diffs$PS - diffs$OL

# Color by difference type
diffs %>% 
  pivot_longer(c('PS-OS', 'PS-OL'), names_to = "Difference", values_to = "val") %>%
  ggplot(aes(x = model, y = val, fill = Difference, alpha = Difference)) + 
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) +
  scale_fill_manual(values = c("PS-OS" = "purple", "PS-OL" = "turquoise")) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1), 
  ) + 
  ylab("Difference in Accuracy") + 
  xlab("") + 
  ggtitle("Difference in Accuracy Across Conditions by Model")
ggsave("figs/overall-diff-v2.pdf", width=6, height=4)

# Color by model, alpha by difference type 
diffs %>% 
  pivot_longer(c('PS-OS', 'PS-OL'), names_to = "Difference", values_to = "val") %>%
  ggplot(aes(x = model, y = val, fill = model, alpha = Difference)) + 
  scale_alpha_manual(values = c(0.25, 1)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  guides(
    fill = "none",
    alpha = guide_legend(override.aes = list(fill = "gray40"))
  ) + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1), 
  ) + 
  ylab("Difference in Accuracy") + 
  xlab("") + 
  ggtitle("Difference in Accuracy Across Conditions by Model")
ggsave("figs/overall-diff.pdf", width=6, height=4)

# Box plots of the overall difference 
temp <- everything %>% 
  filter(data_type == "PS" & train_size == "Small") %>%
  group_by(model, split) %>% 
  summarize(meanf1_ps = round(mean(f1), 3))

# Explore the difference in F1 scores between PS and OS 
diff <- inner_join(temp, table2, by=c("model", "split"))
diff$f1_diff = diff$meanf1_ps - diff$meanf1
diff %>%
  ggplot(aes(x = model, y = f1_diff, fill = model)) + 
  geom_col(alpha = 0.7) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Difference in Accuracy") + 
  xlab("") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1), 
        legend.position = "none"
  ) + 
  ggtitle("Difference in Accuracy from Small Plus Short to Only Short by Model")
ggsave("figs/acc-diff-ps-os-box.pdf", width=6, height=4)

# Explore the difference in F1 scores between PS on OL
diff <- inner_join(temp, table4, by=c("model", "split"))
diff$f1_diff = diff$meanf1_ps - diff$meanf1
diff %>%
  ggplot(aes(x = model, y = f1_diff, fill = model)) + 
  geom_col(alpha = 0.7) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Difference in Accuracy") + 
  xlab("") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 35, vjust = 1, hjust = 1), 
        legend.position = "none"
  ) + 
  ggtitle("Difference in Accuracy from Small Plus Short to Only Short by Model")
ggsave("figs/acc-diff-ps-ol-box.pdf", width=6, height=4)

# F1 by datatype 
everything %>%
  ggplot(aes(x = model, y = f1, fill = factor(data_type, levels=c("OS", "PS", "OL")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Accuracy") + 
  xlab("Model") + 
  labs(fill = "Data Type") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 30, vjust = 1, hjust = 1)) +
  ggtitle("Average Accuracy by Data Type") 
ggsave("figs/acc-by-datatype.pdf")


# Performance by language class
everything %>%
  filter(data_type =="OS") %>% 
  ggplot(aes(x = language_class, y = f1, fill = model)) + 
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Accuracy") + 
  labs(fill="Model")+
  xlab("Language Class") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average Accuracy by Language Class + Model on Only Short") 
ggsave("figs/os-by-class.pdf", width = 15, height = 7)


everything %>%
  filter(data_type =="PS") %>% 
  ggplot(aes(x = language_class, y = f1, fill = model)) + 
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Accuracy") + 
  labs(fill="Model")+
  xlab("Language Class") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average Accuracy by Language Class + Model on Plus Short") 
ggsave("figs/ps-by-class.pdf", width = 15, height = 7)

everything %>%
  filter(data_type =="OL") %>% 
  ggplot(aes(x = language_class, y = f1, fill = model)) + 
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.2, lwd = 0.2) + 
  facet_wrap(~factor(split, levels=c("SR", "SA", "LR", "LA"))) + 
  theme_bw() + 
  ylab("Accuracy") + 
  labs(fill="Model")+
  xlab("Language Class") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average Accuracy by Language Class + Model on Only Long") 
ggsave("figs/ol-by-class.pdf", width = 15, height = 7)

