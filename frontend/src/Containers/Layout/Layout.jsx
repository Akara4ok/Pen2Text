import React from 'react';
import classes from './Layout.scss';
import FileUploader from '@Components/FileUploader/FileUploader';
import TextEditor from '@Components/TextEditor/TextEditor';
import Button from '@Components/Button/Button';
import Backdrop from '@Components/Backdrop/Backdrop';
import Spinner from '@Components/Spinner/Spinner';
import Message from '../../Components/Message/Message';
import { buildHtmlFromResponse } from '../../utils/utils';

let lastTargetEvent;

class Layout extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isDragEnter: false,
            isSendingRequest: false,
            files: [],
            language: 'English',
            networkName: 'Letters',
            plainText: [],
            errorMsgs: [],
        };
    }

    dragEnterHandler = event => {
        event.preventDefault();
        event.stopPropagation();
        const { isDragEnter } = this.state;
        lastTargetEvent = event.target;
        if (!isDragEnter) this.setState({ isDragEnter: true });
    };

    dragEndHandler = event => {
        const { isDragEnter } = this.state;
        if (isDragEnter && lastTargetEvent == event.target) {
            event.preventDefault();
            event.stopPropagation();
            this.setState({ isDragEnter: false });
        }
    };

    onDropHandler = event => {
        const { isDragEnter } = this.state;
        event.preventDefault();
        event.stopPropagation();
        if (isDragEnter) {
            lastTargetEvent = undefined;
            this.setState({ isDragEnter: false });
        }
    };

    dragOverHandler = event => {
        event.preventDefault();
    };

    setFiles = newFiles => {
        this.setState({ files: newFiles });
    };

    setLanguage = lang => {
        this.setState({ language: lang });
    };

    setNetworkName = name => {
        this.setState({ networkName: name });
    };

    sendRequestHandler = event => {
        const { files, language, networkName } = this.state;

        event.preventDefault();
        const formData = new FormData();
        formData.append('language', language);
        formData.append('networkName', networkName);
        for (let index = 0; index < files.length; index++) {
            formData.append('file', files[index]);
        }
        this.setState({ isSendingRequest: true });
        fetch('http://localhost:5000/pen_text', {
            method: 'POST',
            body: formData,
        })
            .then(response => {
                if (response.status >= 200 && response.status <= 299) {
                    return response.json();
                }
                return Promise.reject(response);
                // throw new Error(error ?? 'Undefined error');
            })
            .then(data => {
                console.log(data);
                this.setState({
                    isSendingRequest: false,
                    plainText: data.data.plain_text,
                });
            })
            .catch(response => {
                response.json().then(error => {
                    this.setState({
                        isSendingRequest: false,
                        errorMsgs: error?.errors,
                    });
                });
            });
    };

    onErrorHandlerClick = () => {
        this.setState({ errorMsgs: [] });
    };

    render() {
        const { isDragEnter, isSendingRequest, plainText, errorMsgs } =
            this.state;
        return (
            <div
                className={classes.wrapper}
                onDragEnter={this.dragEnterHandler}
                onDragLeave={this.dragEndHandler}
                onDragOver={this.dragOverHandler}
                onDrop={this.onDropHandler}>
                <div className={classes.title}>Pen2Text</div>
                <div className={classes.editorContainer}>
                    <FileUploader
                        isFileDroping={isDragEnter}
                        setFiles={this.setFiles}
                        setLanguage={this.setLanguage}
                        setNetworkName={this.setNetworkName}
                    />
                    <TextEditor plainText={plainText} />
                </div>
                <Button
                    className={classes.submitButton}
                    onClick={this.sendRequestHandler}>
                    Submit
                </Button>
                {isSendingRequest ? (
                    <Backdrop>
                        <Spinner />
                    </Backdrop>
                ) : null}
                {errorMsgs.length > 0 ? (
                    <Backdrop>
                        <Message onClose={this.onErrorHandlerClick}>
                            {errorMsgs.map((element, index) => (
                                <p key={'error' + index}>{element.message}</p>
                            ))}
                        </Message>
                    </Backdrop>
                ) : null}
            </div>
        );
    }
}

export default Layout;
