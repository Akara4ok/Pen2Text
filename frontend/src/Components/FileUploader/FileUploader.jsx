import React from 'react';
import classes from './FileUploader.scss';
import UploadButton from './UploadButton/UploadButton';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';
import FileView from './FileView/FileView';
import FileList from './FileList/FileList';
import FileItem from './FileList/FileItem/FileItem';
import DragAndDrop from './DragAndDrop/DragAndDrop';
import { FaExchangeAlt } from 'react-icons/fa';
import PenEditor from './PenEditor/PenEditor';
import Backdrop from '@Components/Backdrop/Backdrop';
import Message from '../Message/Message';
class FileUploader extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            currentFileNo: 0,
            drawnFiles: 0,
            files: [],
            isFileViewerMode: true,
            language: 'English',
            networkName: 'Letters',
            errorMsgs: [],
        };
    }

    typeChecker = /^image\/.*|application\/pdf/;
    networkNameDict = {
        English: ['Letters', 'Letters+Numbers', 'All chars'],
        Ukrainian: ['Letters'],
    };

    uploadHandler = event => {
        const uploadedFiles = event.target.files;
        this.updateFiles(uploadedFiles);
    };

    updateFiles = uploadedFiles => {
        const { files, errorMsgs } = this.state;
        let currentFileNo = files.length - 1;
        for (let index = 0; index < uploadedFiles.length; index++) {
            if (!this.typeChecker.test(uploadedFiles[index].type)) {
                errorMsgs.push({
                    message:
                        'The content of file is not correct. Filename: ' +
                        uploadedFiles[index].name +
                        '\n',
                });
                continue;
            }

            currentFileNo++;
            files.push(uploadedFiles[index]);
        }
        this.props.setFiles(files);
        this.setState({
            files,
            errorMsgs,
            currentFileNo,
        });
    };

    getNextFile = () => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        currentFileNo++;
        if (currentFileNo > files.length - 1) {
            currentFileNo = 0;
        }
        this.setState({ currentFileNo: currentFileNo });
        return files[currentFileNo];
    };

    getPreviousFile = () => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        currentFileNo--;
        if (currentFileNo < 0) {
            currentFileNo = files.length - 1;
        }
        this.setState({ currentFileNo: currentFileNo });
        return files[currentFileNo];
    };

    goToSelectedFile = index => {
        this.setState({ currentFileNo: index });
    };

    deleteByIndex = index => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        files.splice(index, 1);
        if (index <= currentFileNo) {
            currentFileNo--;
            if (currentFileNo < 0) {
                currentFileNo = files.length - 1;
            }
        }
        this.props.setFiles(files);
        this.setState({ currentFileNo: currentFileNo, files: [...files] });
    };

    increaseDrawnFiles = () => {
        let { drawnFiles } = this.state;
        this.setState({ drawnFiles: drawnFiles + 1 });
    };

    changeMode = () => {
        let { isFileViewerMode } = this.state;
        isFileViewerMode = !isFileViewerMode;
        this.setState({ isFileViewerMode });
    };

    setFileViewMode = () => {
        this.setState({ isFileViewerMode: true });
    };

    setLanguage = language => {
        this.setState({ language: language });
        this.props.setLanguage(language);
    };

    setNetworkName = name => {
        this.setState({ networkName: name });
        this.props.setNetworkName(name);
    };

    onErrorHandlerClick = () => {
        this.setState({ errorMsgs: [] });
    };

    render() {
        const { isFileDroping } = this.props;
        const {
            files,
            currentFileNo,
            isFileViewerMode,
            drawnFiles,
            language,
            errorMsgs,
        } = this.state;
        return (
            <div className={classes.wrapper}>
                <div className={classes.content}>
                    {isFileViewerMode ? (
                        <FileView
                            previosFile={files[currentFileNo - 1]}
                            currentFile={files[currentFileNo]}
                            nextFile={files[currentFileNo + 1]}
                            getNextFile={this.getNextFile}
                            getPreviousFile={this.getPreviousFile}
                            fileCount={files.length}
                        />
                    ) : (
                        <PenEditor
                            updateFiles={this.updateFiles}
                            increaseDrawnFiles={this.increaseDrawnFiles}
                            drawnFiles={drawnFiles}
                        />
                    )}
                    <div className={classes.listButtonWrapper}>
                        <UploadButton uploadHandler={this.uploadHandler} />
                        <FileList>
                            {files.map((element, index) => (
                                <FileItem
                                    id={element.name + index}
                                    key={element.name + index}
                                    index={index}
                                    goToSelectedFile={this.goToSelectedFile}
                                    deleteByIndex={this.deleteByIndex}
                                    setFileViewMode={this.setFileViewMode}>
                                    {element.name}
                                </FileItem>
                            ))}
                        </FileList>
                        <Button
                            className={classes.buttonStyle}
                            onClick={this.changeMode}>
                            <FaExchangeAlt size={15} /> <span>Pen</span>
                        </Button>
                    </div>
                </div>
                {isFileDroping ? (
                    <DragAndDrop onDrop={this.updateFiles} />
                ) : null}
                <div className={classes.dropdownStyle}>
                    <DropdownList
                        items={['English', 'Ukrainian']}
                        setValue={this.setLanguage}
                    />
                    <DropdownList
                        className={classes.networkName}
                        items={this.networkNameDict[language]}
                        setValue={this.setNetworkName}
                    />
                </div>
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

export default FileUploader;
