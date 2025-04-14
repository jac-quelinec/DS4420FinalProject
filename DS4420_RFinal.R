# packages
library(tidyverse)
library(scales)
library(FNN)
library(knitr)
library(ggplot2)
library(maps)
library(sf)
library(tigris)
library(heatmaply)
library(htmlwidgets)

options(tigris_use_cache = TRUE)

# dataset
df <- read.csv("C:/Users/Avita/Downloads/FoodAccessResearchAtlasData2019.csv")

# key features
features <- c(
  "CensusTract", "State", "PovertyRate", "MedianFamilyIncome", "Urban", 
  "TractKids", "TractSeniors", "TractHUNV", "TractSNAP",
  "lahunvhalfshare", "lapop1share", "lalowi1share",
  "TractWhite", "TractBlack", "TractAsian", "TractNHOPI",
  "TractAIAN", "TractOMultir", "TractHispanic"
)

# preprocessing

df_clean <- df %>%
  select(all_of(features)) %>%
  drop_na()

df_clean_fixed <- df_clean %>%
  mutate(across(-c(CensusTract, State), ~ as.numeric(gsub("[^0-9.]", "", .x))))

# aggregate to state level
df_state <- df_clean_fixed %>%
  group_by(State) %>%
  summarise(across(-CensusTract, mean, na.rm = TRUE), .groups = "drop")

# normalize features
df_norm <- df_state %>%
  mutate(across(-State, ~ rescale(.x, to = c(0, 1))))

# generating interventions training 
set.seed(42)
interventions <- data.frame(
  State = df_norm$State,
  MobileMarket = sample(rep(c(0, 1), length.out = nrow(df_norm))),
  SNAPOutreach = sample(rep(c(0, 1), length.out = nrow(df_norm))),
  GroceryGrant = sample(rep(c(0, 1), length.out = nrow(df_norm)))
)

# KNN similarity matrix
feature_matrix <- df_norm %>%
  column_to_rownames("State") %>%
  as.matrix()

knn_result <- get.knn(feature_matrix, k = 10)

knn_df <- data.frame(
  State = rep(rownames(feature_matrix), each = 10),
  Neighbor = rownames(feature_matrix)[as.vector(knn_result$nn.index)],
  Similarity = as.vector(1 - knn_result$nn.dist)
)

# intervention data
intervention_long <- interventions %>%
  pivot_longer(-State, names_to = "Intervention", values_to = "Applied")

# weighted recommendations according to similarity
recommend_df <- knn_df %>%
  inner_join(intervention_long %>% mutate(State = as.character(State)), by = c("Neighbor" = "State")) %>%
  mutate(WeightedScore = Similarity * Applied)

recommendation <- recommend_df %>%
  group_by(State, Intervention) %>%
  summarise(RecommendationScore = sum(WeightedScore, na.rm = TRUE), .groups = "drop") %>%
  group_by(State) %>%
  slice_max(RecommendationScore, n = 1) %>%
  ungroup()

# map plotting
data("state")
us_states <- map_data("state")

recommendation_named <- recommendation %>%
  mutate(region = tolower(State))

map_data_joined <- us_states %>%
  left_join(recommendation_named, by = "region")

# plot state-level map
ggplot(map_data_joined, aes(x = long, y = lat, group = group, fill = Intervention)) +
  geom_polygon(color = "white", size = 0.2) +
  coord_fixed(1.3) +
  scale_fill_manual(values = c(
    "MobileMarket" = "#FADADD",   
    "SNAPOutreach" = "#FFC75F",  
    "GroceryGrant" = "#FF9671"   
  )) +
  labs(
    title = "Top Recommended Food Access Intervention by State",
    fill = "Intervention"
  ) +
  theme_minimal(base_family = "Helvetica") +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 8),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )



# Method

# 15 sample states
sample_states <- c("Alabama", "California", "New York", "Texas", "Florida",
                   "Illinois", "Georgia", "Michigan", "Ohio", "Pennsylvania",
                   "North Carolina", "Arizona", "Massachusetts", "Washington", "Tennessee")

# similarity values
knn_unique <- knn_df %>%
  group_by(State, Neighbor) %>%
  summarise(Similarity = max(Similarity), .groups = "drop")

# similarity matrix
similarity_subset <- knn_unique %>%
  filter(State %in% sample_states & Neighbor %in% sample_states) %>%
  pivot_wider(
    names_from = Neighbor,
    values_from = Similarity,
    values_fill = list(Similarity = 0)
  ) %>%
  column_to_rownames("State") %>%
  as.matrix()

# heatmap
heatmap_plot <- heatmaply(
  similarity_subset,
  dendrogram = "none",  
  xlab = "Neighbor State",
  ylab = "Focal State",
  main = "Similarity Between 15 Selected States",
  colors = colorRampPalette(c("white", "#FDBCB4", "#E63946"))(50),
  fontsize_row = 10,
  fontsize_col = 10,
  margins = c(70, 100, 40, 20),
  grid_color = "white"
)

# html png
saveWidget(
  widget = heatmap_plot,
  file = "heatmap_15_states.html",
  selfcontained = TRUE
)


