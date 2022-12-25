import React from 'react';
import classes from './Layout.scss';
import FileUploader from '@Components/FileUploader/FileUploader';
import TextEditor from '@Components/TextEditor/TextEditor';
import Button from '@Components/Button/Button';
import Backdrop from '@Components/Backdrop/Backdrop';
import Spinner from '@Components/Spinner/Spinner';

let lastTargetEvent;

class Layout extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isDragEnter: false,
            isSendingRequest: false,
            files: [],
            language: 'English'
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
        this.setState({files: newFiles});
    }

    setLanguage = lang => {
        this.setState({ language: lang });
    }



    sendRequestHandler = (event) => {
        const { files, language } = this.state;
        
        event.preventDefault();
        const formData = new FormData();
        formData.append('language', language);
        for (let index = 0; index < files.length; index++) {
            formData.append('file', files[index]);
        }
        this.setState({isSendingRequest: true});
        fetch('http://localhost:5000/pen_text', {
            method: 'POST',
            body: formData
          })
          .then(response => {
            if (response.status >= 200 && response.status <= 299) {
              return response.json();
            } else {
              return response.json().then(error => {throw new Error(error?.errors ?? "Undefined error")});
              //throw Error(response.statusText);
            }
          })
          .then(data => {
            console.log(data);
            this.setState({isSendingRequest: false});
          })
          .catch(error => {
            console.log(error);
            this.setState({isSendingRequest: false});
          })
        
        
    }

    render() {
        const { isDragEnter, isSendingRequest } = this.state;
        return (
            <div
                className={classes.wrapper}
                onDragEnter={this.dragEnterHandler}
                onDragLeave={this.dragEndHandler}
                onDragOver={this.dragOverHandler}
                onDrop={this.onDropHandler}>
                <div className={classes.title}>Pen2Text</div>
                <div className={classes.editorContainer}>
                    <FileUploader isFileDroping={isDragEnter} setFiles={this.setFiles} setLanguage={this.setLanguage}/>
                    <TextEditor />
                </div>
                <Button className={classes.submitButton} onClick={this.sendRequestHandler}>Submit</Button>
                {
                    isSendingRequest ?
                    <Backdrop><Spinner/></Backdrop> : null
                }
            </div>
        );
    }
}

export default Layout;
