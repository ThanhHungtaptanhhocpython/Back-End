terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }

  # Lưu ý: Với Azure for Students, bạn có thể lưu state cục bộ (local)
  # hoặc cấu hình backend azurerm sau khi tạo storage account đầu tiên.
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# 1. Resource Group chứa toàn bộ tài nguyên
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location

  tags = var.tags
}

# 2. Azure Storage Account (tương đương S3 của AWS) để lưu trữ 100GB Keyframes & Embeddings
resource "azurerm_storage_account" "storage" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = var.storage_tier
  account_replication_type = var.storage_replication_type
  access_tier              = "Hot" # Tối ưu tốc độ đọc ghi cho truy vấn và load dữ liệu

  # Cấu hình bảo mật & hiệu năng
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  https_traffic_only_enabled      = true

  tags = var.tags
}

# 3. Blob Container chứa dữ liệu Video Keyframes (Ảnh .webp / .jpg)
resource "azurerm_storage_container" "keyframes" {
  name                  = "keyframes"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# 4. Blob Container chứa dữ liệu Vector Embeddings (.npy, .parquet, faiss index)
resource "azurerm_storage_container" "embeddings" {
  name                  = "embeddings"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# 5. Blob Container chứa Dữ liệu Từ điển & Metadata (.json, .csv)
resource "azurerm_storage_container" "metadata" {
  name                  = "metadata"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}
