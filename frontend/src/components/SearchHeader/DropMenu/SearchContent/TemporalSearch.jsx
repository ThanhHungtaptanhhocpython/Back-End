import { Form, Input, Button, Switch, InputNumber } from "antd";
import { MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";

const initialValue = {
  query: "",
  cascaded: true,
  topk: 10,
};

const TemporalSearch = ({ setRequire }) => {
  const [form] = Form.useForm();

  const onFinish = (values) => {
    console.log("Received values of form:", values);
    setRequire(values);
  };

  return (
    <Form
      form={form}
      name="dynamic_form_nest_item"
      onFinish={onFinish}
      autoComplete="off"
      style={{ maxWidth: "900px", margin: "0 auto" }}
      initialValues={initialValue}
    >
      <Form.List name="query">
        {(fields, { add, remove }) => (
          <>
            {fields.map(({ key, name, ...restField }) => (
              <div
                key={key}
                style={{
                  display: "flex",
                  alignItems: "center",
                  marginBottom: 10,
                }}
              >
                <Form.Item
                  {...restField}
                  name={[name, "query"]}
                  style={{ flex: 1, marginBottom: 0 }}
                  rules={[{ required: true, message: "Missing query" }]}
                >
                  <Input size="large" placeholder="Input your query" />
                </Form.Item>
                <MinusCircleOutlined
                  onClick={() => remove(name)}
                  style={{
                    fontSize: 18,
                    cursor: "pointer",
                    marginLeft: 8,
                  }}
                />
              </div>
            ))}

            <Form.Item>
              <Button
                type="dashed"
                onClick={() => add()}
                block
                icon={<PlusOutlined />}
                size="large"
              >
                Add query
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>

      <Form.Item
        label={<span style={{ fontSize: 16 }}>Cascaded mode</span>}
        valuePropName="checked"
        name="cascaded"
        initialValue={true}
      >
        <Switch />
      </Form.Item>

      <Form.Item label="Top K" name="topk">
        <InputNumber style={{ width: "120px" }} min="10" defaultValue="10" />
      </Form.Item>

      <Form.Item style={{ marginTop: "20px" }}>
        <Button type="primary" htmlType="submit" block>
          <span style={{ fontSize: 16 }}>Search</span>
        </Button>
      </Form.Item>
    </Form>
  );
};

export default TemporalSearch;
