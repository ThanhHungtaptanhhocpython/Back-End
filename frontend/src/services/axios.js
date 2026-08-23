import axios from 'axios';

const instance = axios.create({
  baseURL: 'https://84e893faad66.ngrok-free.app/users',
});

instance.interceptors.response.use(function (response) {
    return response.data;
  }, function (error) {
    return Promise.reject(error);
  });

export default instance;