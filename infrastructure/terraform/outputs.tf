output "storage_account_name" {
  value       = azurerm_storage_account.storage.name
  description = "Tên Storage Account vừa tạo"
}

output "storage_primary_access_key" {
  value       = azurerm_storage_account.storage.primary_access_key
  description = "Access Key để Backend kết nối tải/đọc dữ liệu"
  sensitive   = true
}

output "storage_connection_string" {
  value       = azurerm_storage_account.storage.primary_connection_string
  description = "Connection string để cấu hình vào file .env"
  sensitive   = true
}

output "keyframes_container_url" {
  value       = "${azurerm_storage_account.storage.primary_blob_endpoint}keyframes"
  description = "URL container chứa 100GB Keyframes"
}

output "embeddings_container_url" {
  value       = "${azurerm_storage_account.storage.primary_blob_endpoint}embeddings"
  description = "URL container chứa Vector Embeddings"
}
