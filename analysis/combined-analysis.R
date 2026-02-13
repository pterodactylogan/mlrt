library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

#################### ANALYZING EVERYTHING TOGETHER ############################

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


all_neural <- load_nn("../neural")
all_ff <- load_ff("../FlexFringe/FlexFringe")
everything <- rbind(all_neural, all_ff)


# Look at size related performance for plus short 
everything %>%
  filter(data_type == "PS") %>%
  filter(train_size != "Large") %>%
  ggplot(aes(x = model, y = f1, fill = factor(train_size, levels = c("Small", "Mid")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~split) + 
  labs(fill="Train Size") + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average F1 by Size + Model on PlusShort") 

# Look at size related performance for only long 
everything %>%
  filter(data_type == "OL") %>%
  filter(train_size != "Large") %>%
  ggplot(aes(x = model, y = f1, fill = factor(train_size, levels = c("Small", "Mid")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~split) + 
  labs(fill="Train Size") + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average F1 by Size + Model on OnlyLong") 

# Accuracy by training size on each of the test sets 
everything %>%
  filter(split == "SR") %>% 
  filter(train_size != "Large") %>%
  ggplot(aes(x = model, y = f1, fill = factor(train_size, levels = c("Small", "Mid")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~factor(data_type, levels=c("OS", "PS", "OL"))) + 
  labs(fill="Train Size") + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) + 
  ggtitle("Performance by Model and Training Size on Short Random Test")

everything %>%
  filter(split == "SA") %>% 
  filter(train_size != "Large") %>%
  ggplot(aes(x = model, y = f1, fill = factor(train_size, levels = c("Small", "Mid")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~factor(data_type, levels=c("OS", "PS", "OL"))) + 
  labs(fill="Train Size") + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) + 
  ggtitle("Performance by Model and Training Size on Short Adversarial Test")

everything %>%
  filter(split == "LR") %>% 
  filter(train_size != "Large") %>%
  ggplot(aes(x = model, y = f1, fill = factor(train_size, levels = c("Small", "Mid")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~factor(data_type, levels=c("OS", "PS", "OL"))) + 
  labs(fill="Train Size") + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) + 
  ggtitle("Performance by Model and Training Size on Long Random Test")


everything %>%
  filter(split == "LA") %>% 
  filter(train_size != "Large") %>%
  ggplot(aes(x = model, y = f1, fill = factor(train_size, levels = c("Small", "Mid")))) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~factor(data_type, levels=c("OS", "PS", "OL"))) + 
  labs(fill="Train Size") + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) + 
  ggtitle("Performance by Model and Training Size on Long Adversarial Test")



####################### CONSIDERING ONLY ONE SIZE ##############################



# Global variable for size
size = "Mid"


# Functions to make the FF and NN outputs compatible bc they didn't for some reason 
cleanup_nn <- function(df, size, type){
  new_df <- df %>% filter(train_set_size == size) %>%
    mutate(data_type = type) %>%
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
           model = network_type
    ) 
  return(new_df)
}

cleanup_ff <- function(df, type){
  new_df <- df %>%
    mutate(split = str_remove(split, "Test")) %>%
    filter(!split %in% c("Train", "Dev")) %>%
    mutate(model = "FF") %>%
    mutate(data_type = type) %>%
    filter(alphabet_size < 64) %>%
    select(-data_size, -ini, -model_path, -data_path, -last_modified)
}


# Load in and cleanup the neural stuff 
neural_os <- read.csv("../neural/onlyshort_evals.csv")
neural_ps <- read.csv("../neural/plusshort_evals.csv")
neural_ol <- read.csv("../neural/standard_evals.csv")
neural_ps_size <- cleanup_nn(neural_ps, size, "PS")
neural_os_size <- cleanup_nn(neural_os, size, "OS")
neural_ol_size <- cleanup_nn(neural_ol, size, "OL")

# Load in and cleanup the FF stuff 
ff_os <- read.csv("../FlexFringe/FlexFringe/models-os/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
full_ps_path <- paste0("../FlexFringe/FlexFringe/models-ps-", tolower(size), "/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
ff_ps_size <- read.csv(full_ps_path)
full_ol_path <- paste0("../FlexFringe/FlexFringe/models-reg-", tolower(size), "/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
ff_ol_size <- read.csv(full_ol_path)
ff_os <- cleanup_ff(ff_os, "OS")
ff_ps_size <- cleanup_ff(ff_ps_size, "PS")
ff_ol_size <- cleanup_ff(ff_ol_size, "OL")

# Make the combined DFs
both_ps <- rbind(ff_ps_size, neural_ps_size)
both_os <- rbind(ff_os, neural_os_size)
both_ol <- rbind(ff_ol_size, neural_ol_size)
all <- rbind(both_ps, both_os, both_ol)



# Plot averaged accuracy by test set for each model on PS
both_ps %>%
  ggplot(aes(x = model, y = f1, fill=model)) + 
  geom_boxplot(alpha = 0.5) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle(paste0("Average F1 for PlusShort Data, size = ", size)) 
ggsave("figs/f1-PS.pdf")
  

# Plot averaged accuracy by test set for each model on OS
both_os %>%
  ggplot(aes(x = model, y = f1, fill = model)) + 
  geom_boxplot(alpha=0.5) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle(paste0("Average F1 for OnlyShort Data, size = ", size)) 
ggsave("figs/f1-OS.pdf")


# Plot averaged accuracy by test set for each model on OL 
both_ol %>%
  ggplot(aes(x = model, y = f1, fill = model)) + 
  geom_boxplot(alpha=0.5) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle(paste0("Average F1 for OnlyLong Data, size = ", size)) 
ggsave("figs/f1-OL.pdf")

# Plot them together 
all %>%
  ggplot(aes(x = model, y = f1, fill = data_type)) +
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average F1 by Data Type") 
ggsave("figs/f1-by-datatype.pdf")

# Get the performance in a table 
summarized <- all %>% group_by(model, data_type, split) %>%
  summarise(meanf1 = mean(f1))


# Now plotting by language class, like Adil's Figure 2. First PS 
both_ps %>% 
  ggplot(aes(x = language_class, y = f1, fill = model)) + 
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average F1 by Language Class + Model on PlusShort") 
ggsave("figs/ps-by-class.pdf", width = 15, height = 7)

# Same thing but on only short 
both_os %>% 
  ggplot(aes(x = language_class, y = f1, fill = model)) + 
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average F1 by Language Class + Model on OnlyShort") 
ggsave("figs/os-by-class.pdf", width = 15, height = 7)

# Finally, same thing but only long 
both_ol %>% ggplot(aes(x = language_class, y = f1, fill = model)) + 
  geom_boxplot(alpha=0.6) + 
  facet_wrap(~split) + 
  theme_bw() + 
  ylab("F1 Score") + 
  xlab("Model") + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 67, vjust = 1, hjust = 1)) +
  ggtitle("Average F1 by Language Class + Model on OnlyLong") 
ggsave("figs/ol-by-class.pdf", width = 15, height = 7)
  







