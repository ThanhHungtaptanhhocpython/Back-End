import React from "react";
import { UploadOutlined } from "@ant-design/icons";
import { Button, message, Upload, Form, InputNumber, Switch } from "antd";

const ImageSearch = ({ setRequire }) => {
  const [form] = Form.useForm();

  const onFinish = (values) => {
    console.log("Form values:", values); 
    setRequire(values);

    message.success("Parameters set successfully!");
  };

  const uploadProps = {
    beforeUpload: (file) => {
      form.setFieldsValue({
        image: file,
      });

      message.success(`${file.name} selected successfully`);
      return false;
    },
  };

  const initialValue = {
    image: null,
    topk: 10,
    clip: true,
    clipv2: false,
    faiss_index: "default",
  };

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onFinish}
      name="image_search"
      initialValues={initialValue}
    >
      <Form.Item name="image">
        <Upload {...uploadProps} maxCount={1}>
          <Button size="large" icon={<UploadOutlined />}>Click to Upload</Button>
        </Upload>
      </Form.Item>

      <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
        <Form.Item label="Top K" name="topk">
          <InputNumber style={{ width: "120px" }} min={10} />
        </Form.Item>

        <Form.Item label="Clip" valuePropName="checked" name="clip">
          <Switch />
        </Form.Item>

        <Form.Item label="Clipv2" valuePropName="checked" name="clipv2">
          <Switch />
        </Form.Item>

        {/* Bạn có thể để hidden field luôn giữ faiss_index */}
        <Form.Item name="faiss_index" initialValue="default" hidden>
          <input type="hidden" />
        </Form.Item>
      </div>

      <Form.Item style={{ marginTop: "20px" }}>
        <Button type="primary" htmlType="submit" block>
          <span style={{ fontSize: 16 }}>Search</span>
        </Button>
      </Form.Item>
    </Form>
  );
};

export default ImageSearch;
