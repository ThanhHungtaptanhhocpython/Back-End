import React, { useState } from "react";
import { Form, Input, Button, InputNumber, Space } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";

const initialValue = {
  query: "",
  ocr: "",
  objectDetection: "",
  topk: 10,
};
const isDisabled = true; 

const OcrAndOdSearch = ({ setRequire }) => {
  const [listOd, setListOd] = useState([]); // Lưu các ô được chọn
  const letters = ["a", "b", "c", "d", "e", "f", "g"];

  const onFinish = (values) => {
    console.log("Received values of form:", values);
    setRequire(values);
  };

  // Xử lý khi click vào nút trong grid
  const handleInputClickOd = (name) => {
    if (!listOd.includes(name)) {
      setListOd([...listOd, name]);
    }
  };

  // Xóa ô khỏi danh sách
  const handleRemove = (name) => {
    setListOd(listOd.filter((item) => item !== name));
  };

  return (
    <Form
      name="ocr_od_search"
      layout="vertical"
      onFinish={onFinish}
      initialValues={initialValue}
      autoComplete="off"
    >
      <Form.Item name="query">
        <Input
          size="large"
          placeholder="Input your query (optional)"
          style={{ width: "100%" }}
        />
      </Form.Item>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "16px",
        }}
      >
        <div style={{ flex: 1 }}>
          <Form.Item name="ocr" label="OCR">
            <Input
              size="large"
              placeholder="Enter OCR text"
              style={{ width: "80%" }}
            />
          </Form.Item>
          {/* <Form.Item name="objectDetection" label="Object Detection">
            <Input
              size="large"
              placeholder="Enter object class (e.g., dog, cat)"
              style={{ width: "80%" }}
            />
          </Form.Item> */}
          <div style={{ marginBottom: "8px" }}>Object and number</div>
          <Form.List name="objectDetection">
            {(fields, { add, remove }) => (
              <>
                {fields.map(({ key, name, ...restField }) => (
                  <Space
                    key={key}
                    style={{ display: "flex", marginBottom: 8 }}
                    align="baseline"
                  >
                    <Form.Item
                      {...restField}
                      name={[name, "class"]}
                      rules={[{ required: true, message: "Missing class" }]}
                    >
                      <Input placeholder="Input class" />
                    </Form.Item>
                    <Form.Item
                      {...restField}
                      name={[name, "number"]}
                      rules={[{ required: true, message: "Missing number" }]}
                    >
                      <Input placeholder="Input number" />
                    </Form.Item>
                    <MinusCircleOutlined onClick={() => remove(name)} />
                  </Space>
                ))}
                <Form.Item>
                  <Button
                    type="dashed"
                    onClick={() => add()}
                    block
                    icon={<PlusOutlined />}
                  >
                    Add Object
                  </Button>
                </Form.Item>
              </>
            )}
          </Form.List>

          <Form.Item label="Top K" name="topk">
            <InputNumber
              style={{ width: "120px" }}
              min="10"
              defaultValue="10"
            />
          </Form.Item>
        </div>
        <div
          style={{
            flex: 1,
            opacity: isDisabled ? 0.5 : 1,
            pointerEvents: isDisabled ? "none" : "auto",
          }}
        >
          <div style={{ marginBottom: "8px" }}>
            Bounding box: Choose the class for each available position
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(7, 1fr)",
              gap: 0,
            }}
          >
            {Array.from({ length: 49 }, (_, index) => {
              const row = Math.floor(index / 7) + 1;
              const col = index % 7;
              const name = `${row}${letters[col]}`;

              return (
                <Form.Item key={name} style={{ marginBottom: 0 }}>
                  <Button
                    type="dashed"
                    block
                    icon={<PlusOutlined />}
                    onClick={() => handleInputClickOd(name)}
                  >
                    {name.toUpperCase()}
                  </Button>
                </Form.Item>
              );
            })}
          </div>

          <div style={{ marginTop: "16px" }}>
            {listOd.map((name) => (
              <Form.Item
                key={name}
                label={`Input for ${name.toUpperCase()}`}
                style={{ marginBottom: 0 }}
                name={["bbox", name]}
                initialValue=""
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  <Input placeholder={`Enter value for ${name}`} />
                  <Button
                    type="text"
                    icon={<MinusCircleOutlined />}
                    onClick={() => handleRemove(name)}
                  />
                </div>
              </Form.Item>
            ))}
          </div>
        </div>
      </div>
      <Form.Item style={{ marginTop: "20px" }}>
        <Button type="primary" htmlType="submit" block>
          <span style={{ fontSize: 16 }}>Search</span>
        </Button>
      </Form.Item>
    </Form>
  );
};

export default OcrAndOdSearch;
