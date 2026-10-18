import http from 'k6/http';
import { check, sleep } from 'k6';

// 200-second stress-test stages scaling to 10,000 - 20,000 RPS
export const options = {
  stages: [
    { duration: '30s', target: 2000 },   // Warm-up to 2,000 RPS
    { duration: '60s', target: 10000 },  // Ramp to 10,000 RPS (NFR2 threshold)
    { duration: '80s', target: 20000 },  // Peak load: 20,000 RPS
    { duration: '30s', target: 0 },      // Ramp-down
  ],
  thresholds: {
    // SLAs: Strictly < 10ms overall latency (NFR1)
    http_req_duration: ['p(95)<5', 'p(99)<10'],
    http_req_failed: ['rate<0.01'], // 99% reliability target
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8001';

export default function () {
  const threatScore = Math.random();
  const payload = JSON.stringify({
    metadata: {
      method: 'POST',
      path: '/api/v1/checkout',
      content_type: 'application/json',
      client_ip: '192.168.1.100',
      client_role: 'customer',
      authorization: 'Bearer token-test-k6',
      headers: {
        'x-origin': 'k6-load-client',
      },
    },
    payload: {
      email: 'customer-k6@example.com',
      password: 'do-not-leak-this-password',
      card_number: '4111 1111 1111 1111',
      cart_total: 129.95,
      tax: 10.4,
      items: 4,
    },
    threat_score: threatScore,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(`${BASE_URL}/v1/ingest`, payload, params);

  check(res, {
    'status is 202 Accepted': (r) => r.status === 202,
    'context vector exists': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.unified_context_vector && body.unified_context_vector.length > 0;
      } catch (e) {
        return false;
      }
    },
    'protobuf size > 0': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.protobuf_bytes_size > 0;
      } catch (e) {
        return false;
      }
    },
    'latency under SLA (<10ms)': (r) => r.timings.duration < 10,
  });
}
