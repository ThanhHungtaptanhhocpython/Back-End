variable "resource_group_name" {
  type        = string
  description = "Tên Resource Group trên Azure"
  default     = "rg-aic-backend"
}

variable "location" {
  type        = string
  description = "Khu vực đặt dữ liệu Azure (khuyến nghị southeastasia - Singapore để ping về VN thấp nhất)"
  default     = "southeastasia"
}

variable "storage_account_name" {
  type        = string
  description = "Tên Storage Account (chỉ chứa chữ thường và số, độ dài 3-24 ký tự, phải là duy nhất toàn cầu)"
}

variable "storage_tier" {
  type        = string
  description = "Tier của Storage Account (Standard hoặc Premium)"
  default     = "Standard"
}

variable "storage_replication_type" {
  type        = string
  description = "Cơ chế sao lưu dữ liệu (LRS - Locally Redundant Storage tiết kiệm chi phí nhất cho Azure Student)"
  default     = "LRS"
}

variable "tags" {
  type        = map(string)
  description = "Tags gán cho tài nguyên"
  default = {
    Project     = "AI-Challenge-2025"
    Environment = "Development"
    Owner       = "Student"
  }
}
