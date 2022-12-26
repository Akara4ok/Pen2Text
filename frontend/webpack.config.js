const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
    mode: 'development',
    entry: './src/index',
    resolve: {
        modules: [__dirname, 'src', 'node_modules'],
        extensions: ['*', '.js', '.jsx', '.tsx', '.ts'],
        fallback: {
            crypto: false,
            fs: false,
            path: false,
        },
        alias: {
            '@global': path.join(__dirname, 'src/global'),
            '@Components': path.join(__dirname, 'src/Components'),
            '@Containers': path.join(__dirname, 'src/Containers'),
        },
    },
    output: {
        filename: 'bundle.[fullhash].js',
        path: path.resolve(__dirname, 'build'),
        clean: true,
    },
    module: {
        rules: [
            {
                test: /\.js$/i,
                exclude: /node_modules/,
                include: path.resolve(__dirname, 'src/'),
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env'],
                    },
                },
            },
            {
                test: /\.jsx?$/,
                exclude: /node_modules/,
                loader: require.resolve('babel-loader'),
            },
            {
                test: /\.css$/i,
                use: ['style-loader', 'css-loader'],
            },
            {
                test: /\.s[ac]ss$/i,
                use: [
                    'style-loader',
                    {
                        loader: 'css-loader',
                        options: {
                            modules: {
                                mode: 'local',
                                localIdentName:
                                    '[name]__[local]___[hash:base64:5]',
                            },
                        },
                    },
                    'postcss-loader',
                    'sass-loader',
                ],
            },
            {
                test: /\.(jpg|jpeg|png|svg|gif)$/i,
                type: 'asset/resource',
            },
            {
                test: /\.(woff|woff2|eot|ttf|otf)$/i,
                type: 'asset/resource',
            },
        ],
    },
    devtool: 'inline-source-map',
    plugins: [
        new HtmlWebpackPlugin({
            title: 'Pen2Text',
            favicon: './static/logo/favicon.ico',
            template: path.resolve(__dirname, './src/index.html'),
        }),
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: 'static',
                    noErrorOnMissing: true,
                },
            ],
        }),
    ],
    devServer: {
        static: path.resolve(__dirname, 'build/static'),
        hot: true,
        port: 3000,
        compress: true,
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods':
                'GET, POST, PUT, DELETE, PATCH, OPTIONS',
            'Access-Control-Allow-Headers':
                'X-Requested-With, content-type, Authorization',
        },
    },
};
