import { Form, InputNumber, Select, Switch, Input, Button } from "antd";

const initialValue = {
  query: "",
  topk: 10,
  clip: true,
  clipv2: false,
};

const QASearch = ({ setRequire }) => {
  const onFinish = (values) => {
    if (values) {
      console.log("Received values of form:", values);
      setRequire(values);
    }
  };
  return (
    <>
      <Form
        layout="vertical"
        onFinish={onFinish}
        name="single_text_search"
        initialValues={initialValue}
      >
        <Form.Item name="query">
          <Input
            size="large"
            placeholder="Input your query"
            style={{ width: "100%" }}
          />
        </Form.Item>
        <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
          <Form.Item label="Top K" name="topk">
            <InputNumber
              style={{ width: "120px" }}
              min="10"
              defaultValue="10"
            />
          </Form.Item>

          <Form.Item
            label="Clip"
            valuePropName="checked"
            name="clip"
            defaultValue="true"
          >
            <Switch />
          </Form.Item>

          <Form.Item label="Clipv2" valuePropName="checked" name="clipv2">
            <Switch />
          </Form.Item>
        </div>

        <Form.Item style={{ marginTop: "20px" }}>
          <Button type="primary" htmlType="submit" block>
            <span style={{ fontSize: 16 }}>Search</span>
          </Button>
        </Form.Item>
      </Form>
    </>
  );
};

export default QASearch;
