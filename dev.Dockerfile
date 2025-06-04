FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ENV PATH="./node_modules/.bin:$PATH"
EXPOSE 5173
RUN apk add --no-cache curl

CMD ["npm", "run", "dev"]
