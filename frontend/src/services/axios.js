import axios from 'axios';

const baseURL = (import.meta.env.VITE_SEARCH_API_BASE_URL || 'http://127.0.0.1:3000') + '/users';

const instance = axios.create({
  baseURL: baseURL,
});

instance.interceptors.response.use(function (response) {
    return response.data;
  }, function (error) {
    return Promise.reject(error);
  });

export default instance;