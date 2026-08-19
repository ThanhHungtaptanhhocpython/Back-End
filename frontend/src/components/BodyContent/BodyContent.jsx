import React, { useState, useRef, useEffect } from "react";
import SearchHeader from "../SearchHeader/SearchHeader";
import SearchResults from "../SearchResults/SearchResults";
import { Tabs, ConfigProvider } from "antd";
import HistoryButton from "../HistoryButton/HistoryButton";
import SubmitButton from "../SubmitButton/SubmitButton";
import { Layout } from "antd";
const { Header, Sider, Content } = Layout;

const BodyContent = () => {
  const [tabsData, setTabsData] = useState([]);
  const [activeKey, setActiveKey] = useState(0);
  const [items, setItems] = useState([]);
  const newTabIndex = useRef(0);

  useEffect(() => {
    console.log("tabsData: ", tabsData);
  }, [tabsData]);


  useEffect(() => {
    add();
  }, []);

  const onChange = (newActiveKey) => {
    setActiveKey(newActiveKey);
  };

  const add = () => {
    const key = `newTab${newTabIndex.current}`;
    const label = `New Tab ${newTabIndex.current + 1}`;
    const newTab = {
      key,
      label,
      searchType: "Single Text Search",
      require: {},
      result: {},
    };
    setTabsData((prev) => [...prev, newTab]);
    setActiveKey(key);
    newTabIndex.current += 1;
  };

  const remove = (targetKey) => {
    let newActiveKey = activeKey;
    let lastIndex = -1;

    items.forEach((item, i) => {
      if (item.key === targetKey) {
        lastIndex = i - 1;
      }
    });

    const newPanes = items.filter((item) => item.key !== targetKey);

    if (newPanes.length === 0) {
      setItems([]);
      setActiveKey(null);
      return;
    }

    if (newActiveKey === targetKey) {
      if (lastIndex >= 0) {
        newActiveKey = newPanes[lastIndex].key;
      } else {
        newActiveKey = newPanes[0].key;
      }
    }

    setItems(newPanes);
    setActiveKey(newActiveKey);
  };

  const onEdit = (targetKey, action) => {
    if (action === "add") {
      add();
    } else {
      remove(targetKey);
    }
  };
  const updateTab = (key, field, value) => {
    setTabsData((prev) =>
      prev.map((tab) => (tab.key === key ? { ...tab, [field]: value } : tab))
    );
  };

  return (
    <div>
      <ConfigProvider
        theme={{
          components: {
            Tabs: {
              horizontalMargin: 0,
            },
          },
        }}
      >
        <Tabs
          type="editable-card"
          onChange={onChange}
          activeKey={activeKey}
          onEdit={onEdit}
          items={tabsData.map((tab) => ({
            label: tab.label,
            key: tab.key,
            children: (
              <Layout>
                <ConfigProvider
                  theme={{
                    components: {
                      Layout: {
                        siderBg: "#607B8B",
                      },
                    },
                  }}
                >
                  <Sider width={250}>
                    <div style={{ padding: "20px" }}>
                      <HistoryButton />
                      <SubmitButton result={tab.result} searchType={tab.searchType} />
                    </div>
                  </Sider>
                </ConfigProvider>
                <Layout style={{ background: "#cbd7e2" }}>
                  <Header
                    style={{
                      padding: "0 16px",
                      background: "#cbd7e2",
                      height: "auto",
                    }}
                  >
                    <div>
                      {/* {console.log(tab.key)} */}
                      <SearchHeader
                        setSearchType={(val) =>
                          updateTab(tab.key, "searchType", val)
                        }
                        setRequire={(val) => updateTab(tab.key, "require", val)}
                      />
                    </div>
                  </Header>
                  <Content
                    style={{
                      margin: "10px 16px",
                      background: "#cbd7e2",
                    }}
                  >
                    <SearchResults
                      searchType={tab.searchType}
                      require={tab.require}
                      setResult={(val) => updateTab(tab.key, "result", val)}
                      result={tab.result}
                    />
                  </Content>
                </Layout>
              </Layout>
            ),
          }))}
        />
      </ConfigProvider>
    </div>
  );
};

export default BodyContent;
