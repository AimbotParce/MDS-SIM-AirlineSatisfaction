required_packages <- c(
    "rmarkdown", "FactoMineR", "cluster", "missMDA",
    "corrplot", "pROC", "ggplot2", "PRROC", "car"
) # Add more packages as needed

install_if_missing <- function(packages) {
    for (pkg in packages) {
        if (!requireNamespace(pkg, quietly = TRUE)) {
            install.packages(pkg, repos = "https://cloud.r-project.org/")
        }
    }
}

install_if_missing(required_packages)
