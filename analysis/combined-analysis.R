library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

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
neural_ps_small <- cleanup_nn(neural_ps, "Small", "PS")
neural_os_small <- cleanup_nn(neural_os, "Small", "OS")
neural_ol_small <- cleanup_nn(neural_ol, "Small", "OL")

# Load in and cleanup the FF stuff 
ff_os <- read.csv("../FlexFringe/FlexFringe/models-os/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
ff_ps_small <- read.csv("../FlexFringe/FlexFringe/models-ps-small/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
ff_ol_small <- read.csv("../FlexFringe/FlexFringe/models-reg-small/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
ff_os <- cleanup_ff(ff_os, "OS")
ff_ps_small <- cleanup_ff(ff_ps_small, "PS")
ff_ol_small <- cleanup_ff(ff_ol_small, "OL")

# Make the combined DFs
both_ps <- rbind(ff_ps_small, neural_ps_small)
both_os <- rbind(ff_os, neural_os_small)
both_ol <- rbind(ff_ol_small, neural_ol_small)
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
  ggtitle("Average F1 for PlusShort Data") 
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
  ggtitle("Average F1 for OnlyShort Data") 
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
  ggtitle("Average F1 for OnlyLong Data") 
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
  







