# ==============================================================================
# Azure Storage Outputs
# ==============================================================================
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
  description = "URL container chứa 100GB-200GB Keyframes"
}

output "embeddings_container_url" {
  value       = "${azurerm_storage_account.storage.primary_blob_endpoint}embeddings"
  description = "URL container chứa Vector Embeddings"
}

output "metadata_container_url" {
  value       = "${azurerm_storage_account.storage.primary_blob_endpoint}metadata"
  description = "URL container chứa Metadata & Dictionaries"
}

output "videos_container_url" {
  value       = "${azurerm_storage_account.storage.primary_blob_endpoint}videos"
  description = "URL container chứa Video gốc (.mp4)"
}

output "models_container_url" {
  value       = "${azurerm_storage_account.storage.primary_blob_endpoint}models"
  description = "URL container chứa Model Checkpoints (BEiT-3, CLIP, BLIP)"
}


# ==============================================================================
# IAM User Credentials cho đồng đội
# ==============================================================================
output "teammate_username" {
  value       = azuread_user.teammate_user.user_principal_name
  description = "Tài khoản email để teammate đăng nhập vào portal.azure.com"
}

output "teammate_password" {
  value       = random_password.teammate_pwd.result
  description = "Mật khẩu để teammate đăng nhập vào portal.azure.com"
  sensitive   = true
}

# ==============================================================================
# Elasticsearch VM Outputs (Public IP & Credentials)
# ==============================================================================
output "elasticsearch_public_ip" {
  value       = azurerm_public_ip.es_ip.ip_address
  description = "Địa chỉ IP Public của máy ảo Elasticsearch"
}

output "elasticsearch_url_for_env" {
  value       = "http://elastic:${random_password.es_password.result}@${azurerm_public_ip.es_ip.ip_address}:9200"
  description = "URL kết nối Elasticsearch để điền vào .env máy Local và Google Colab"
  sensitive   = true
}

output "elasticsearch_password" {
  value       = random_password.es_password.result
  description = "Mật khẩu tài khoản elastic"
  sensitive   = true
}

output "vm_ssh_command" {
  value       = "ssh ${var.vm_admin_username}@${azurerm_public_ip.es_ip.ip_address}"
  description = "Lệnh SSH vào quản trị máy ảo VM"
}

output "vm_admin_password" {
  value       = random_password.vm_admin_password.result
  description = "Mật khẩu SSH của user azureuser"
  sensitive   = true
}
