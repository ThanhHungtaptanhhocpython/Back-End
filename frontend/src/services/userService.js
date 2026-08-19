import axios from "./axios";

const singleTextSearch = async (params) => {
  try {
    const response = await axios.post(
      "/singletextsearch",
      {
        query: params.query,
        topk: params.topk,
        clip: params.clip,
        clipv2: params.clipv2,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error in single text search:", error);
    throw error;
  }
};

const QASearch = async (params) => {
  try {
    const response = await axios.post(
      "/qnasearch",
      {
        query: params.query,
        topk: params.topk,
        clip: params.clip,
        clipv2: params.clipv2,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error in single text search:", error);
    throw error;
  }
};

const OcrAndOdSearch = async (params) => {
  try {
    const response = await axios.post(
      "/ocrandodsearch",
      {
        query: params.query,
        topk: params.topk,
        ocr: params.ocr,
        od: params.od,
        bbox: params.bbox,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error in single text search:", error);
    throw error;
  }
};

const imageSearch = async (params) => {
  try {
    const formData = new FormData();
    console.log("Params in image search:", params.image);
    if (params.image != null) {
      formData.append("image", params.image?.file || params.image);
    }
    else {
      formData.append("image", null);
    }
    formData.append("topk", params.topk);
    formData.append("faiss_index", params.faiss_index);
    formData.append("clip", params.clip);
    formData.append("clipv2", params.clipv2);

    const response = await axios.post("/imagesearch", formData);

    return response.data;
  } catch (error) {
    console.error("Error in image search:", error);
    throw error;
  }
};

const temporalSearch = async (params) => {
  try {
    const response = await axios.post(
      "/temporalsearch",
      {
        query: params.query,
        topk: params.topk,
        cascaded: params.cascaded,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error in single text search:", error);
    throw error;
  }
};

export { singleTextSearch, QASearch, OcrAndOdSearch, imageSearch, temporalSearch };
