terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.48"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

provider "azuread" {}
provider "random" {}

provider "azurerm" {
  storage_use_azuread = true
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Lấy thông tin Client / Tenant ID hiện tại
data "azurerm_client_config" "current" {}

# ==============================================================================
# 1. Resource Group chứa toàn bộ tài nguyên
# ==============================================================================
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location

  tags = var.tags
}

# ==============================================================================
# 2. Azure Storage Account (tương đương S3 của AWS) để lưu trữ 100GB-200GB Keyframes & Embeddings
# ==============================================================================
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
  shared_access_key_enabled       = true

  tags = var.tags
}

# 3. Blob Containers
resource "azurerm_storage_container" "keyframes" {
  name                  = "keyframes"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "embeddings" {
  name                  = "embeddings"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "metadata" {
  name                  = "metadata"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "videos" {
  name                  = "videos"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}


# ==============================================================================
# 3. IAM: Tự động tạo Tài khoản User (Username & Password) cho bạn đồng đội đăng nhập
# ==============================================================================
data "azuread_domains" "default" {
  only_initial = true
}

resource "random_password" "teammate_pwd" {
  length           = 16
  special          = true
  override_special = "!#$&*-_=+@."
}

resource "azuread_user" "teammate_user" {
  user_principal_name   = "teammate@${data.azuread_domains.default.domains[0].domain_name}"
  display_name          = "AI Challenge Teammate"
  mail_nickname         = "teammate"
  password              = random_password.teammate_pwd.result
  force_password_change = false
}

resource "azurerm_role_assignment" "teammate_storage_reader" {
  scope                = azurerm_storage_account.storage.id
  role_definition_name = "Reader"
  principal_id         = azuread_user.teammate_user.object_id
}

resource "azurerm_role_assignment" "teammate_storage_data_contributor" {
  scope                = azurerm_storage_account.storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azuread_user.teammate_user.object_id
}

# ==============================================================================
# 4. MẠNG & BẢO MẬT (VNet, Subnet, Public IP, Firewall NSG) CHO MÁY ẢO ELASTICSEARCH
# ==============================================================================
resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-aic-backend"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  tags = var.tags
}

resource "azurerm_subnet" "subnet" {
  name                 = "subnet-aic-backend"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_public_ip" "es_ip" {
  name                = "pip-elasticsearch"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = var.tags
}

resource "azurerm_network_security_group" "es_nsg" {
  name                = "nsg-elasticsearch"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  # Cổng 22: SSH quản trị
  security_rule {
    name                       = "Allow-SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  # Cổng 9200: Elasticsearch API (kết nối từ Google Colab & Local)
  security_rule {
    name                       = "Allow-Elasticsearch-9200"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "9200"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = var.tags
}

resource "azurerm_network_interface" "es_nic" {
  name                = "nic-elasticsearch"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.es_ip.id
  }

  tags = var.tags
}

resource "azurerm_network_interface_security_group_association" "es_nic_nsg" {
  network_interface_id      = azurerm_network_interface.es_nic.id
  network_security_group_id = azurerm_network_security_group.es_nsg.id
}

# ==============================================================================
# 5. MÁY ẢO LINUX UBUNTU + TỰ ĐỘNG CHẠY DOCKER ELASTICSEARCH
# ==============================================================================
resource "random_password" "vm_admin_password" {
  length           = 16
  special          = true
  override_special = "!#$&*-_=+@."
}

resource "random_password" "es_password" {
  length           = 16
  special          = false # Chữ và số cho mật khẩu ES kết nối URL dễ dàng
}

resource "azurerm_linux_virtual_machine" "es_vm" {
  name                            = "vm-elasticsearch"
  resource_group_name             = azurerm_resource_group.rg.name
  location                        = azurerm_resource_group.rg.location
  size                            = var.vm_size # Standard_B2s (2 vCPU, 4GB RAM)
  admin_username                  = var.vm_admin_username
  admin_password                  = random_password.vm_admin_password.result
  disable_password_authentication = false
  network_interface_ids           = [azurerm_network_interface.es_nic.id]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = 30
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }

  # Script tự động chạy khi khởi động máy ảo: Cài Docker + bật Elasticsearch 8.13
  custom_data = base64encode(<<-EOF
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker

    # Khởi chạy container Elasticsearch với mật khẩu và 2GB RAM
    docker run -d \
      --name elasticsearch \
      -p 9200:9200 \
      -e "discovery.type=single-node" \
      -e "xpack.security.enabled=true" \
      -e "ELASTIC_PASSWORD=${random_password.es_password.result}" \
      -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
      -v es_data:/usr/share/elasticsearch/data \
      --restart always \
      docker.elastic.co/elasticsearch/elasticsearch:8.13.0
  EOF
  )

  tags = var.tags
}
