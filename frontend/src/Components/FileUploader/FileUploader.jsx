import React from 'react';
import classes from './FileUploader.scss';
import UploadButton from './UploadButton/UploadButton';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';

class FileUploader extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        return (
            <div className={classes.wrapper}>
                <div>Upload files</div>
                <UploadButton>upload</UploadButton>
                <DropdownList className={classes.dropdownStyle} />
                <Button className={classes.buttonStyle}>To Pen</Button>
            </div>
        );
    }
}

export default FileUploader;
